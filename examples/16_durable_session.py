"""Persist Core Conversation and Audit state across process invocations."""

import argparse
import asyncio
import os
from pathlib import Path

from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.storage import JsonlSessionStore
from ejagent.tools import FunctionToolExecutor

AGENT_ID = "durable-session-agent"
SYSTEM_PROMPT = (
    "Preserve user-provided facts exactly and answer only from durable history."
)


def harness(store: JsonlSessionStore) -> AgentHarness:
    return AgentHarness(
        agent_id=AGENT_ID,
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=FunctionToolExecutor(),
        store=store,
        initial_messages=(SystemMessage(SYSTEM_PROMPT),),
    )


async def record(store: JsonlSessionStore) -> None:
    agent = harness(store)
    async with agent:
        outcome = await agent.run(
            "Remember this exact project code and reply only ACK: CORE-2048"
        )
        print(f"record response: {outcome.result.output}")
    print(f"committed revision: {agent.revision}")


async def resume(store: JsonlSessionStore) -> None:
    if await store.load(AGENT_ID) is None:
        raise RuntimeError("run the record command before resume")
    agent = harness(store)
    async with agent:
        outcome = await agent.run("Return only the exact project code stored before.")
        print(f"resume response: {outcome.result.output}")
    print(f"audited runs: {len(await store.load_audit(AGENT_ID))}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("record", "resume"))
    parser.add_argument(
        "--session-dir",
        default=os.getenv("EJAGENT_SESSION_DIR", ".ejagent-sessions"),
    )
    args = parser.parse_args()
    store = JsonlSessionStore(Path(args.session_dir))
    if args.command == "record":
        await record(store)
    else:
        await resume(store)


if __name__ == "__main__":
    asyncio.run(main())
