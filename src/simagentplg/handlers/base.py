from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from inspect import Parameter, signature
from typing import Any

from simagentplg.agent.cancellation import (
    CancellationSource,
    CancellationToken,
)
from simagentplg.agent.types import StepOutcome, ToolProgressReporter
from simagentplg.handlers.definition import (
    ToolDefinition,
    ToolDefinitionError,
    ToolDefinitionInput,
    ToolEffect,
    ToolSchema,
    normalize_tool_definition,
    normalize_tool_definitions,
)
from simagentplg.middleware.base import ToolCallContext


class UnknownToolError(KeyError):
    """Raised when a handler is asked to execute an unknown tool."""


class BaseHandler(ABC):
    """Interface implemented by reusable groups of related tools."""

    @property
    @abstractmethod
    def tools(self) -> Sequence[ToolSchema]:
        """Return tool definitions in OpenAI function-calling format."""

    @property
    def tool_definitions(self) -> Sequence[ToolDefinition]:
        """Return canonical definitions for legacy BaseHandler subclasses."""

        return normalize_tool_definitions(self.tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tool_definitions)

    async def startup(self) -> None:
        """Initialize optional external resources."""

    async def shutdown(self) -> None:
        """Release optional external resources."""

    async def on_task_start(self) -> None:
        """Prepare handler state for one new agent task."""

    def can_handle(self, tool_name: str) -> bool:
        return tool_name in self.tool_names

    def tool_effect(self, tool_name: str) -> ToolEffect:
        """Return the conservative execution effect for one registered tool."""

        for tool in self.tool_definitions:
            if tool.name == tool_name:
                return tool.effect
        raise UnknownToolError(f"unknown tool {tool_name!r}")

    async def execute(self, context: ToolCallContext) -> StepOutcome:
        """Execute a runtime context while adapting legacy dispatch methods."""

        kwargs: dict[str, Any] = {
            "cancellation": context.cancellation,
        }
        if _accepts_keyword(self.dispatch, "progress"):
            kwargs["progress"] = context.progress
        return await self.dispatch(
            context.tool_name,
            context.arguments,
            **kwargs,
        )

    @abstractmethod
    async def dispatch(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        """Execute a registered tool with optional run cancellation."""


class MethodToolHandler(BaseHandler):
    """Handler that maps tool names to async ``do_<tool_name>`` methods."""

    def __init__(
        self,
        tools: Sequence[ToolDefinitionInput],
        *,
        tool_effects: Mapping[str, ToolEffect] | None = None,
    ) -> None:
        definitions = list(normalize_tool_definitions(tools))
        names = tuple(tool.name for tool in definitions)
        if len(names) != len(set(names)):
            raise ToolDefinitionError("handler contains duplicate tool names")
        effects = dict(tool_effects or {})
        unknown = effects.keys() - set(names)
        if unknown:
            names_text = ", ".join(sorted(unknown))
            raise ToolDefinitionError(
                f"tool effects reference unknown tool(s): {names_text}"
            )
        for name, effect in effects.items():
            if not isinstance(effect, ToolEffect):
                raise ToolDefinitionError(
                    f"tool effect for {name!r} must be a ToolEffect"
                )
        self._tool_definitions = tuple(
            normalize_tool_definition(
                tool,
                effect=effects.get(tool.name),
            )
            for tool in definitions
        )

    @property
    def tools(self) -> Sequence[ToolSchema]:
        return tuple(tool.to_openai_tool() for tool in self._tool_definitions)

    @property
    def tool_definitions(self) -> Sequence[ToolDefinition]:
        return self._tool_definitions

    def tool_effect(self, tool_name: str) -> ToolEffect:
        if not self.can_handle(tool_name):
            raise UnknownToolError(f"unknown tool {tool_name!r}")
        return next(
            tool.effect for tool in self._tool_definitions if tool.name == tool_name
        )

    async def dispatch(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        if not self.can_handle(tool_name):
            raise UnknownToolError(f"unknown tool {tool_name!r}")

        method = getattr(self, f"do_{tool_name}", None)
        if method is None:
            raise ToolDefinitionError(
                f"{type(self).__name__} must define do_{tool_name}()"
            )

        token = cancellation or CancellationSource().token
        kwargs: dict[str, Any] = {"cancellation": token}
        if _accepts_keyword(method, "progress"):
            kwargs["progress"] = progress
        outcome = await method(dict(arguments), **kwargs)
        if not isinstance(outcome, StepOutcome):
            raise TypeError(
                f"do_{tool_name}() must return StepOutcome, "
                f"got {type(outcome).__name__}"
            )
        return outcome


def _accepts_keyword(callable_: Any, keyword: str) -> bool:
    """Return whether a callable accepts one explicit or variadic keyword."""

    parameters = signature(callable_).parameters
    parameter = parameters.get(keyword)
    if parameter is not None and parameter.kind in {
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    }:
        return True
    return any(item.kind is Parameter.VAR_KEYWORD for item in parameters.values())
