"""Persist a Session to disk and resume it in a separate invocation."""

import argparse
import asyncio
import os
from pathlib import Path

from ejagent import (
    BaseAgent,
    JsonlSessionStorage,
    ModelConfig,
    OpenAIModelAdapter,
    SessionRecorder,
)

SESSION_ID = "durable-session-demo"
AGENT_ID = "durable-session-agent"
SYSTEM_PROMPT = (
    "Preserve user-provided facts exactly and answer only from durable "
    "conversation history."
)


async def record(storage: JsonlSessionStorage) -> None:
    recorder = SessionRecorder(session_id=SESSION_ID, storage=storage)
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id=AGENT_ID,
        system_prompt=SYSTEM_PROMPT,
        event_sink=recorder,
    )
    try:
        result = await agent.run(
            task=(
                "Remember this exact project code and reply only ACK: SIM-SESSION-2048"
            )
        )
        print(f"record response: {result.output}")
    finally:
        await agent.shutdown()

    saved = await storage.load(SESSION_ID)
    if saved is None:
        raise RuntimeError("durable Session was not saved")
    print(f"saved runs: {len(saved.runs)}")


async def resume(storage: JsonlSessionStorage) -> None:
    saved = await storage.load(SESSION_ID)
    if saved is None:
        raise RuntimeError("run the record command before resume")

    recorder = SessionRecorder(session_id=SESSION_ID, storage=storage)
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id=AGENT_ID,
        system_prompt=SYSTEM_PROMPT,
        event_sink=recorder,
    )
    agent.restore_session(saved)
    try:
        result = await agent.run(
            task="Return only the exact project code stored previously."
        )
        print(f"resume response: {result.output}")
    finally:
        await agent.shutdown()

    updated = await storage.load(SESSION_ID)
    if updated is None:
        raise RuntimeError("resumed Session was not saved")
    print(f"saved runs after resume: {len(updated.runs)}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("record", "resume"))
    parser.add_argument(
        "--session-dir",
        default=os.getenv("EJAGENT_SESSION_DIR", ".ejagent-sessions"),
    )
    args = parser.parse_args()
    storage = JsonlSessionStorage(Path(args.session_dir))
    if args.command == "record":
        await record(storage)
    else:
        await resume(storage)


if __name__ == "__main__":
    asyncio.run(main())
