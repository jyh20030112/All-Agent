from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ejagent.contracts.control import CancellationToken, RunCancelledError
from ejagent.contracts.json import JsonObject
from ejagent.contracts.messages import ToolCall
from ejagent.contracts.tools import (
    ToolDefinition,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutor,
    ToolSemantics,
)
from ejagent.plugins.mcp.mcp_manager import McpServerManager


class McpManager(Protocol):
    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    def get_openai_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, tool_name: str, args: dict[str, object]) -> str: ...


class McpToolExecutor(ToolExecutor):
    """Expose an MCP manager through the provider-neutral ToolExecutor seam."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        manager: McpManager | None = None,
        semantics: Mapping[str, ToolSemantics] | None = None,
    ) -> None:
        if (config_path is None) == (manager is None):
            raise ValueError("provide exactly one of config_path or manager")
        if manager is None:
            assert config_path is not None
            self._manager: McpManager = McpServerManager(config_path)
        else:
            self._manager = manager
        self._semantics = dict(semantics or {})
        if not all(
            isinstance(item, ToolSemantics) for item in self._semantics.values()
        ):
            raise TypeError("MCP semantics values must be ToolSemantics")
        self._definitions: tuple[ToolDefinition, ...] = ()
        self._started = False

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return self._definitions

    async def start(self) -> None:
        if self._started:
            return
        await self._manager.startup()
        try:
            definitions = tuple(
                self._definition_from_openai(item)
                for item in self._manager.get_openai_tools()
            )
            names = [definition.name for definition in definitions]
            if len(names) != len(set(names)):
                raise ValueError("MCP manager returned duplicate tool names")
            unknown = self._semantics.keys() - set(names)
            if unknown:
                raise ValueError(
                    "MCP semantics reference unknown tools: "
                    + ", ".join(sorted(unknown))
                )
            self._definitions = definitions
        except BaseException:
            await self._manager.shutdown()
            raise
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        try:
            await self._manager.shutdown()
        finally:
            self._definitions = ()
            self._started = False

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        if not self._started:
            raise ToolExecutionError("MCP ToolExecutor is not started")
        if call.name not in {definition.name for definition in self._definitions}:
            raise ToolExecutionError(f"unknown MCP tool {call.name!r}")
        try:
            result = await cancellation.run(
                self._manager.call_tool(
                    call.name,
                    {key: value for key, value in call.arguments.items()},
                )
            )
        except RunCancelledError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"MCP tool {call.name!r} failed: {exc}") from exc
        return ToolExecutionResult(result)

    def _definition_from_openai(self, value: Any) -> ToolDefinition:
        if not isinstance(value, Mapping) or value.get("type") != "function":
            raise ValueError("MCP tool must use the OpenAI function shape")
        function = value.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("MCP tool must contain a function object")
        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters", {})
        if not isinstance(name, str):
            raise ValueError("MCP function name must be text")
        if description is not None and not isinstance(description, str):
            raise ValueError("MCP function description must be text or null")
        if not isinstance(parameters, Mapping):
            raise ValueError("MCP function parameters must be an object")
        return ToolDefinition(
            name=name,
            description=description,
            input_schema=self._json_object(parameters),
            semantics=self._semantics.get(name, ToolSemantics()),
        )

    @staticmethod
    def _json_object(value: Mapping[str, Any]) -> JsonObject:
        return dict(value)
