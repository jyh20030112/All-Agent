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
    """Load, connect, and route tools across configured MCP services."""

    def __init__(self, path: str | Path):
        """Initialize the manager.

        Args:
            path: MCP configuration JSON path.
        """
        self.path = Path(path)
        self._clients_by_service: dict[str, Any] = {}
        self._tool_routes: dict[str, _McpToolRoute] = {}
        self._openai_tools: list[dict[str, Any]] = []
        self._exit_stack: AsyncExitStack | None = None

    async def startup(self) -> None:
        """Connect configured MCP services and index their tools.

        One service failure is logged and does not block other services.
        """
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
            mcp_configs = MCPConfig.from_dict(json.load(config_file))
            logger.info(
                "Loaded %d MCP service config(s)",
                len(mcp_configs.mcpServers),
            )
            for service_name, server_model in mcp_configs.mcpServers.items():
                service_stack = AsyncExitStack()
                try:
                    logger.info("Connecting MCP service %s", service_name)
                    mcp_client = await service_stack.enter_async_context(
                        Client({service_name: server_model.model_dump()})
                    )
                    tools = await mcp_client.list_tools()
                    self._register_service_tools(service_name, mcp_client, tools)
                    stack.push_async_callback(service_stack.aclose)
                    logger.info(
                        "MCP service %s connected with %d tool(s)",
                        service_name,
                        len(tools),
                    )
                except Exception as e:
                    await service_stack.aclose()
                    logger.error(
                        "Failed to connect MCP service %s: %s",
                        service_name,
                        e,
                    )
        self._exit_stack = stack
        logger.info(
            "MCP server manager started with %d online service(s)",
            len(self._clients_by_service),
        )

    async def shutdown(self) -> None:
        """Close all MCP service connections."""
        logger.info("Shutting down MCP server manager")
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        for service_name in self._clients_by_service:
            logger.info("MCP service %s disconnected", service_name)
        self._clients_by_service.clear()
        self._tool_routes.clear()
        self._openai_tools.clear()
        logger.info("MCP server manager stopped")

    async def call_tool(self, tool_name: str, args: dict[str, object]) -> str:
        """Call a routed MCP tool.

        Args:
            tool_name: Prefixed tool name, such as "playwright__browser_navigate".
            args: Tool arguments.

        Returns:
            String representation of the tool result.

        Raises:
            ValueError: If the tool is not registered.
        """
        try:
            route = self._tool_routes[tool_name]
        except KeyError as exc:
            raise ValueError(f"unknown MCP tool: {tool_name}") from exc

        result = await route.client.call_tool(route.raw_name, args)
        return str(result)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return connected service tools in OpenAI tool format.

        Returns:
            OpenAI tools.
        """
        logger.info("Returning %d OpenAI MCP tool(s)", len(self._openai_tools))
        return list(self._openai_tools)

    def _register_service_tools(
        self,
        service_name: str,
        client: Any,
        tools: list[Any],
    ) -> None:
        routes: dict[str, _McpToolRoute] = {}
        openai_tools: list[dict[str, Any]] = []

        for tool in tools:
            prefixed_name = f"{service_name}__{tool.name}"
            if prefixed_name in self._tool_routes or prefixed_name in routes:
                raise ValueError(f"duplicate MCP tool {prefixed_name!r}")
            routes[prefixed_name] = _McpToolRoute(
                raw_name=tool.name,
                client=client,
            )
            openai_tools.append(
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
        self._openai_tools.extend(openai_tools)
