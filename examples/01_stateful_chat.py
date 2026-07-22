"""Plain chat with stateful conversation memory."""

import asyncio

from ejagent import BaseAgent, ModelConfig, OpenAIModelAdapter


async def main() -> None:
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="tutor",
        system_prompt="You are a concise Python tutor.",
    )

    try:
        first = await agent.runtime(
            task="Remember that my preferred language is Python."
        )
        print(f"First response: {first}")

        second = await agent.runtime(task="Which programming language do I prefer?")
        print(f"Memory response: {second}")

        agent.reset()
        third = await agent.runtime(task="Which programming language do I prefer?")
        print(f"After reset: {third}")
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
