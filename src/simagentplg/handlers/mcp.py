from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from simagentplg.agent.cancellation import (
    CancellationSource,
    CancellationToken,
)
from simagentplg.agent.types import StepOutcome, ToolProgressReporter
from simagentplg.handlers.base import (
    BaseHandler,
    ToolDefinitionError,
    ToolEffect,
    ToolSchema,
    UnknownToolError,
)
from simagentplg.plugins.mcp.mcp_manager import McpServerManager


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
        self._tools: tuple[ToolSchema, ...] = ()
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
        return self._tools

    async def startup(self) -> None:
        if self._started:
            return
        await self.manager.startup()
        try:
            self._tools = tuple(self.manager.get_openai_tools())
            unknown = self._tool_effects.keys() - set(self.tool_names)
            if unknown:
                names_text = ", ".join(sorted(unknown))
                raise ToolDefinitionError(
                    f"tool effects reference unknown MCP tool(s): {names_text}"
                )
        except Exception:
            self._tools = ()
            await self.manager.shutdown()
            raise
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        await self.manager.shutdown()
        self._tools = ()
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
        return self._tool_effects.get(tool_name, ToolEffect.SIDE_EFFECTING)
