"""Render real provider Thinking and Text deltas from Harness events."""

import asyncio

from ejagent import (
    AgentEvent,
    AgentFinished,
    AssistantTextDelta,
    AssistantThinkingDelta,
    BaseAgent,
    MessageCompleted,
    ModelConfig,
    OpenAIModelAdapter,
)


class StreamingConsoleSink:
    def __init__(self) -> None:
        self.received_delta = False
        self.in_thinking = False

    async def emit(self, event: AgentEvent) -> None:
        payload = event.payload
        if isinstance(payload, AssistantThinkingDelta):
            if not self.in_thinking:
                self.in_thinking = True
                print("[thinking] ", end="", flush=True)
            print(payload.delta, end="", flush=True)
        elif isinstance(payload, AssistantTextDelta):
            if self.in_thinking:
                self.in_thinking = False
                print("\n[answer] ", end="", flush=True)
            self.received_delta = True
            print(payload.delta, end="", flush=True)
        elif isinstance(payload, MessageCompleted) and self.received_delta:
            print()
        elif isinstance(payload, AgentFinished):
            print(f"\nfinished: {payload.result.status} / {payload.result.stop_reason}")


async def main() -> None:
    sink = StreamingConsoleSink()
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="streaming-demo",
        system_prompt="Answer concisely and stream the response normally.",
        event_sink=sink,
    )

    try:
        result = await agent.run(task="你有什么能力")
        if not result.succeeded:
            result.raise_for_status()
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
