import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ejagent.tools._mcp_manager import McpServerManager


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(
        self,
        tool_name: str,
        args: dict[str, object],
    ) -> str:
        self.calls.append((tool_name, args))
        return f"called:{tool_name}"


def fake_tool(
    name: str,
    *,
    description: str = "demo tool",
    schema: dict[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


class McpServerManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_tool_uses_prebuilt_route(self) -> None:
        manager = McpServerManager(Path("unused.json"))
        client = FakeMcpClient()
        manager._register_service_tools(  # type: ignore[arg-type]
            "demo",
            client,
            [fake_tool("lookup")],
        )

        result = await manager.call_tool("demo__lookup", {"query": "value"})

        self.assertEqual(result, "called:lookup")
        self.assertEqual(client.calls, [("lookup", {"query": "value"})])

    async def test_unknown_tool_does_not_guess_from_prefix(self) -> None:
        manager = McpServerManager(Path("unused.json"))
        client = FakeMcpClient()
        manager._register_service_tools(  # type: ignore[arg-type]
            "demo",
            client,
            [fake_tool("lookup")],
        )

        with self.assertRaisesRegex(ValueError, "unknown MCP tool"):
            await manager.call_tool("demo__missing", {})

        self.assertEqual(client.calls, [])

    async def test_get_openai_tools_returns_prebuilt_copy(self) -> None:
        manager = McpServerManager(Path("unused.json"))
        manager._register_service_tools(  # type: ignore[arg-type]
            "demo",
            FakeMcpClient(),
            [
                fake_tool(
                    "lookup",
                    description="Lookup a value.",
                    schema={"type": "object"},
                )
            ],
        )

        tools = manager.get_openai_tools()
        tools.append({"type": "function", "function": {"name": "extra"}})

        self.assertEqual(
            manager.get_openai_tools(),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "demo__lookup",
                        "description": "Lookup a value.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

    async def test_duplicate_prefixed_tool_does_not_register_partial_routes(
        self,
    ) -> None:
        manager = McpServerManager(Path("unused.json"))

        with self.assertRaisesRegex(ValueError, "duplicate MCP tool"):
            manager._register_service_tools(  # type: ignore[arg-type]
                "demo",
                FakeMcpClient(),
                [fake_tool("lookup"), fake_tool("lookup")],
            )

        self.assertEqual(manager.get_openai_tools(), [])


if __name__ == "__main__":
    unittest.main()
