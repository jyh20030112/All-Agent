"""Run one task through the Anthropic Messages Provider adapter."""

import asyncio

from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import AnthropicConfig, AnthropicModelPort
from ejagent.tools import FunctionToolExecutor


async def main() -> None:
    harness = AgentHarness(
        agent_id="anthropic-assistant",
        model=AnthropicModelPort(AnthropicConfig.from_env()),
        tools=FunctionToolExecutor(),
        initial_messages=(SystemMessage("Answer precisely."),),
    )

    async with harness:
        outcome = await harness.run("Explain the ModelPort boundary in one sentence.")
        print(outcome.result.output)


if __name__ == "__main__":
    asyncio.run(main())
