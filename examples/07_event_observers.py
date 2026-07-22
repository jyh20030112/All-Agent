"""Observe a real provider-backed agent through multiple event sinks."""

import asyncio
from collections import Counter

from ejagent import (
    AgentEvent,
    AgentEventKind,
    BaseAgent,
    CompositeAgentEventSink,
    ModelConfig,
    OpenAIModelAdapter,
)


class ConsoleEventSink:
    async def emit(self, event: AgentEvent) -> None:
        print(f"event #{event.sequence}: {event.kind} run={event.run_id[:8]}")


class EventMetricsSink:
    def __init__(self) -> None:
        self.counts: Counter[AgentEventKind] = Counter()

    async def emit(self, event: AgentEvent) -> None:
        self.counts[event.kind] += 1


async def main() -> None:
    metrics = EventMetricsSink()
    event_sink = CompositeAgentEventSink([ConsoleEventSink(), metrics])
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="event-demo",
        system_prompt="Answer concisely in one paragraph.",
        event_sink=event_sink,
    )

    try:
        result = await agent.run(
            task="Explain why lifecycle events are useful in an Agent Harness."
        )
        print(f"result: {result.status} / {result.stop_reason}")
        print(f"output: {result.output}")
        print(
            "metrics:",
            {kind.value: count for kind, count in metrics.counts.items()},
        )
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
