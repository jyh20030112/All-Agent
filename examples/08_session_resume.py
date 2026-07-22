"""Persist a real conversation and resume it in a new agent instance."""

import asyncio

from ejagent import (
    BaseAgent,
    MemorySessionStorage,
    ModelConfig,
    OpenAIModelAdapter,
    SessionRecorder,
)


async def main() -> None:
    storage = MemorySessionStorage()
    recorder = SessionRecorder(session_id="resume-demo", storage=storage)

    first_agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="session-demo",
        system_prompt=(
            "Preserve user-provided facts exactly. Answer only from the "
            "conversation and do not invent implementation details."
        ),
        event_sink=recorder,
    )
    try:
        first = await first_agent.run(
            task=(
                "Store this exact statement and reply only ACK: EJAgent Core "
                "uses lifecycle events to build Sessions without coupling "
                "persistence to the Agent Loop."
            )
        )
        print(f"first response: {first.output}")
    finally:
        await first_agent.shutdown()

    saved = await recorder.load()
    if saved is None:
        raise RuntimeError("session was not saved")

    resumed_agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="session-demo",
        system_prompt=(
            "Preserve user-provided facts exactly. Answer only from the "
            "conversation and do not invent implementation details."
        ),
        event_sink=recorder,
    )
    resumed_agent.reset(saved.messages)
    try:
        resumed = await resumed_agent.run(
            task="Repeat the exact stored EJAgent Core statement."
        )
        print(f"resumed response: {resumed.output}")
    finally:
        await resumed_agent.shutdown()

    updated = await recorder.load()
    if updated is None:
        raise RuntimeError("resumed session was not saved")

    print(f"saved runs: {len(updated.runs)}")
    print(f"message roles: {[message['role'] for message in updated.messages]}")


if __name__ == "__main__":
    asyncio.run(main())
