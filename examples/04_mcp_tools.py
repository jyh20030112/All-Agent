"""Expose tools from an MCP server to an AgentHarness."""

import asyncio
from pathlib import Path

from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import McpToolExecutor

MCP_CONFIG = Path(__file__).with_name("mcp_config.json")


async def main() -> None:
    harness = AgentHarness(
        agent_id="browser",
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=McpToolExecutor(MCP_CONFIG),
        initial_messages=(
            SystemMessage(
                "Use MCP tools to inspect the requested page, then answer with "
                "the page title and relevant result summary."
            ),
        ),
    )

    async with harness:
        outcome = await harness.run("Open https://baidu.com and report the page title.")
        print(outcome.result.output)


if __name__ == "__main__":
    asyncio.run(main())
