from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum
from inspect import Parameter, signature
from typing import Any

from simagentplg.agent.cancellation import (
    CancellationSource,
    CancellationToken,
)
from simagentplg.agent.types import StepOutcome, ToolProgressReporter
from simagentplg.middleware.base import ToolCallContext

ToolSchema = dict[str, Any]


class ToolEffect(StrEnum):
    """Declared side-effect class used by the Core tool scheduler."""

    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"


class ToolDefinitionError(ValueError):
    """Raised when a handler exposes an invalid tool definition."""


class UnknownToolError(KeyError):
    """Raised when a handler is asked to execute an unknown tool."""


class BaseHandler(ABC):
    """Interface implemented by reusable groups of related tools."""

    @property
    @abstractmethod
    def tools(self) -> Sequence[ToolSchema]:
        """Return tool definitions in OpenAI function-calling format."""

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tool_name(tool) for tool in self.tools)

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

        return ToolEffect.SIDE_EFFECTING

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

    @staticmethod
    def _tool_name(tool: ToolSchema) -> str:
        try:
            tool_type = tool["type"]
            function = tool["function"]
            name = function["name"]
        except (KeyError, TypeError) as exc:
            raise ToolDefinitionError(
                "tool must contain type and function.name"
            ) from exc

        if tool_type != "function" or not isinstance(name, str) or not name:
            raise ToolDefinitionError("only named function tools are supported")
        return name


class MethodToolHandler(BaseHandler):
    """Handler that maps tool names to async ``do_<tool_name>`` methods."""

    def __init__(
        self,
        tools: Sequence[ToolSchema],
        *,
        tool_effects: Mapping[str, ToolEffect] | None = None,
    ) -> None:
        self._tools = tuple(tools)
        names = self.tool_names
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
        self._tool_effects = effects

    @property
    def tools(self) -> Sequence[ToolSchema]:
        return self._tools

    def tool_effect(self, tool_name: str) -> ToolEffect:
        if not self.can_handle(tool_name):
            raise UnknownToolError(f"unknown tool {tool_name!r}")
        return self._tool_effects.get(tool_name, ToolEffect.SIDE_EFFECTING)

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
