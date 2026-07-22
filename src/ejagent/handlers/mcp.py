from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ejagent.agent.cancellation import (
    CancellationSource,
    CancellationToken,
)
from ejagent.agent.types import StepOutcome, ToolProgressReporter
from ejagent.handlers.base import (
    BaseHandler,
    UnknownToolError,
)
from ejagent.handlers.definition import (
    ToolDefinition,
    ToolDefinitionError,
    ToolEffect,
    ToolSchema,
    normalize_tool_definition,
    normalize_tool_definitions,
)
from ejagent.plugins.mcp.mcp_manager import McpServerManager


class McpToolHandler(BaseHandler):
    """Expose tools from configured MCP servers through the handler API."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        manager: Any | None = None,
        tool_effects: Mapping[str, ToolEffect] | None = None,
    ) -> None:
        if config_path is None and manager is None:
            raise ValueError("config_path is required when manager is not provided")
        if config_path is not None and manager is not None:
            raise ValueError("provide either config_path or manager, not both")
        if manager is not None:
            self.manager = manager
        else:
            assert config_path is not None
            self.manager = McpServerManager(config_path)
        self._tool_definitions: tuple[ToolDefinition, ...] = ()
        effects = dict(tool_effects or {})
        for name, effect in effects.items():
            if not isinstance(effect, ToolEffect):
                raise ToolDefinitionError(
                    f"tool effect for {name!r} must be a ToolEffect"
                )
        self._tool_effects = effects
        self._started = False

    @property
    def tools(self) -> Sequence[ToolSchema]:
        return tuple(tool.to_openai_tool() for tool in self._tool_definitions)

    @property
    def tool_definitions(self) -> Sequence[ToolDefinition]:
        return self._tool_definitions

    async def startup(self) -> None:
        if self._started:
            return
        await self.manager.startup()
        try:
            definitions = normalize_tool_definitions(self.manager.get_openai_tools())
            self._tool_definitions = definitions
            unknown = self._tool_effects.keys() - set(self.tool_names)
            if unknown:
                names_text = ", ".join(sorted(unknown))
                raise ToolDefinitionError(
                    f"tool effects reference unknown MCP tool(s): {names_text}"
                )
            self._tool_definitions = tuple(
                normalize_tool_definition(
                    tool,
                    effect=self._tool_effects.get(tool.name),
                )
                for tool in definitions
            )
        except Exception:
            self._tool_definitions = ()
            await self.manager.shutdown()
            raise
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        await self.manager.shutdown()
        self._tool_definitions = ()
        self._started = False

    async def dispatch(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        if not self.can_handle(tool_name):
            raise UnknownToolError(f"unknown MCP tool {tool_name!r}")
        token = cancellation or CancellationSource().token
        result = await token.run(self.manager.call_tool(tool_name, dict(arguments)))
        return StepOutcome(result)

    def tool_effect(self, tool_name: str) -> ToolEffect:
        if not self.can_handle(tool_name):
            raise UnknownToolError(f"unknown MCP tool {tool_name!r}")
        return next(
            tool.effect for tool in self._tool_definitions if tool.name == tool_name
        )
