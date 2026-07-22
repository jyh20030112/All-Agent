"""Load a local release-note skill by explicit name."""

import asyncio
from pathlib import Path

from ejagent import BaseAgent, ModelConfig, OpenAIModelAdapter

SKILLS_DIR = Path(__file__).with_name("skills")


async def main() -> None:
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="release-writer",
        system_prompt=(
            "Use the explicitly loaded local skill to complete the task. "
            "Return the final deliverable directly."
        ),
        skills_dir=SKILLS_DIR,
    )

    try:
        result = await agent.runtime(
            task=(
                "$release_notes Write release notes for EJAgent Core 0.2.4. Changes: "
                "added an orchestrator, structured run results, and runtime "
                "policy controls."
            )
        )
        print(result)
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
