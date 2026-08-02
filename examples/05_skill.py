"""Project a local release-note skill into one Run's ContextView."""

import asyncio
from pathlib import Path

from ejagent.context import SkillsContextPipeline
from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor

SKILLS_DIR = Path(__file__).with_name("skills")


async def main() -> None:
    harness = AgentHarness(
        agent_id="release-writer",
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=FunctionToolExecutor(),
        context=SkillsContextPipeline(SKILLS_DIR),
        initial_messages=(
            SystemMessage(
                "Use the explicitly loaded local skill and return the final "
                "deliverable directly."
            ),
        ),
    )

    async with harness:
        outcome = await harness.run(
            "$release_notes Write release notes for EJAgent Core 0.6.0. "
            "Changes: added a runtime kernel, lifecycle harness, and durable store."
        )
        print(outcome.result.output)


if __name__ == "__main__":
    asyncio.run(main())
