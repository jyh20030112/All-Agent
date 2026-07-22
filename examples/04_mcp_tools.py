"""Expose tools from an MCP server to an agent."""

import asyncio
from pathlib import Path

from ejagent import (
    BaseAgent,
    McpToolHandler,
    ModelConfig,
    OpenAIModelAdapter,
)

MCP_CONFIG = Path(__file__).with_name("mcp_config.json")


async def main() -> None:
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="browser",
        system_prompt=(
            "Use MCP tools to inspect the requested page, then answer with the "
            "page title and relevant result summary."
        ),
        handlers=[McpToolHandler(MCP_CONFIG)],
    )

    try:
        result = await agent.runtime(
            task="Open https://baidu.com and report the page title."
        )
        print(result)
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
