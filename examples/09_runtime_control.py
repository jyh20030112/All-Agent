"""Abort a real provider request, wait for idle, and reuse the agent."""

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

from ejagent import (
    AgentEvent,
    BaseAgent,
    CancellationToken,
    ModelConfig,
    ModelStreamEvent,
    OpenAIModelAdapter,
)


class ObservableOpenAIModelAdapter(OpenAIModelAdapter):
    """Expose when a real provider request is about to start."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.request_started = asyncio.Event()

    async def stream(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.request_started.set()
        async for event in super().stream(
            context,
            cancellation=cancellation,
        ):
            yield event


class ConsoleEventSink:
    async def emit(self, event: AgentEvent) -> None:
        print(f"event #{event.sequence}: {event.kind}")


async def main() -> None:
    abort_delay = float(os.getenv("HARNESS_ABORT_DELAY", "0.2"))
    model = ObservableOpenAIModelAdapter(ModelConfig.from_env())
    agent = BaseAgent(
        model,
        agent_id="runtime-control-demo",
        event_sink=ConsoleEventSink(),
    )

    try:
        run = asyncio.create_task(
            agent.run(
                task=(
                    "Write a detailed 2000-word technical essay about Agent "
                    "Harness architecture, runtime control, and persistence."
                )
            )
        )
        await model.request_started.wait()
        await asyncio.sleep(abort_delay)

        accepted = agent.abort("cancelled by the runtime-control example")
        await agent.wait_for_idle()
        first_result = await run

        print(f"abort accepted: {accepted}")
        print(f"first result: {first_result.status} / {first_result.stop_reason}")

        reused = await agent.run(task="Reply with exactly: the agent is reusable")
        print(f"reused result: {reused.output}")
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
