"""Resume a committed Conversation in a new AgentHarness instance."""

import asyncio

from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness, MemorySessionStore
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor

AGENT_ID = "session-demo"
SYSTEM_PROMPT = (
    "Preserve user-provided facts exactly. Answer only from the conversation."
)


async def main() -> None:
    store = MemorySessionStore()
    first = AgentHarness(
        agent_id=AGENT_ID,
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=FunctionToolExecutor(),
        store=store,
        initial_messages=(SystemMessage(SYSTEM_PROMPT),),
    )
    async with first:
        outcome = await first.run("Remember this exact code: CORE-2048")
        print(f"first response: {outcome.result.output}")

    resumed = AgentHarness(
        agent_id=AGENT_ID,
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=FunctionToolExecutor(),
        store=store,
        initial_messages=(SystemMessage(SYSTEM_PROMPT),),
    )
    async with resumed:
        outcome = await resumed.run("Return only the exact stored code.")
        print(f"resumed response: {outcome.result.output}")

    print(f"committed revision: {resumed.revision}")
    print(f"audited runs: {len(await store.load_audit(AGENT_ID))}")


if __name__ == "__main__":
    asyncio.run(main())
