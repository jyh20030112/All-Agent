from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ejagent.logger import get_logger

logger = get_logger(name="MCP")


@dataclass(frozen=True, slots=True)
class _McpToolRoute:
    raw_name: str
    client: Any


class McpServerManager:
    """Connect configured MCP services and route namespaced Tool calls."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._clients_by_service: dict[str, Any] = {}
        self._tool_routes: dict[str, _McpToolRoute] = {}
        self._openai_tools: list[dict[str, Any]] = []
        self._exit_stack: AsyncExitStack | None = None

    async def startup(self) -> None:
        """Connect every configured service that can start successfully."""

        logger.info("Starting MCP server manager")
        try:
            from fastmcp import Client
            from fastmcp.mcp_config import MCPConfig
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MCP integration requires optional dependencies; "
                "install ejagent-core[mcp]"
            ) from exc

        stack = AsyncExitStack()
        with self.path.open(encoding="utf-8") as config_file:
            configs = MCPConfig.from_dict(json.load(config_file))
            for service_name, server_model in configs.mcpServers.items():
                service_stack = AsyncExitStack()
                try:
                    client = await service_stack.enter_async_context(
                        Client({service_name: server_model.model_dump()})
                    )
                    tools = await client.list_tools()
                    self._register_service_tools(service_name, client, tools)
                    stack.push_async_callback(service_stack.aclose)
                except Exception as exc:
                    await service_stack.aclose()
                    logger.error(
                        "Failed to connect MCP service %s: %s", service_name, exc
                    )
        self._exit_stack = stack

    async def shutdown(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._clients_by_service.clear()
        self._tool_routes.clear()
        self._openai_tools.clear()

    async def call_tool(self, tool_name: str, args: dict[str, object]) -> str:
        try:
            route = self._tool_routes[tool_name]
        except KeyError as exc:
            raise ValueError(f"unknown MCP tool: {tool_name}") from exc
        result = await route.client.call_tool(route.raw_name, args)
        return str(result)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return list(self._openai_tools)

    def _register_service_tools(
        self,
        service_name: str,
        client: Any,
        tools: list[Any],
    ) -> None:
        routes: dict[str, _McpToolRoute] = {}
        definitions: list[dict[str, Any]] = []
        for tool in tools:
            prefixed_name = f"{service_name}__{tool.name}"
            if prefixed_name in self._tool_routes or prefixed_name in routes:
                raise ValueError(f"duplicate MCP tool {prefixed_name!r}")
            routes[prefixed_name] = _McpToolRoute(tool.name, client)
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": prefixed_name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
            )
        self._clients_by_service[service_name] = client
        self._tool_routes.update(routes)
        self._openai_tools.extend(definitions)
