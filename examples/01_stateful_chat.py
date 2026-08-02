"""Run two tasks against one stateful AgentHarness."""

import asyncio

from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor


async def main() -> None:
    harness = AgentHarness(
        agent_id="tutor",
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=FunctionToolExecutor(),
        initial_messages=(SystemMessage("You are a concise Python tutor."),),
    )

    async with harness:
        first = await harness.run("Remember that my preferred language is Python.")
        print(f"First response: {first.result.output}")

        second = await harness.run("Which programming language do I prefer?")
        print(f"Memory response: {second.result.output}")


if __name__ == "__main__":
    asyncio.run(main())
