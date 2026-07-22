import asyncio
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from ejagent import (
    AgentContextBuilder,
    AgentEvent,
    AgentFinished,
    AgentStarted,
    AgentState,
    AssistantMessage,
    AssistantTextDelta,
    AssistantThinkingDelta,
    BaseAgent,
    CancellationToken,
    CompositeAgentEventSink,
    MemorySessionStorage,
    MessageCompleted,
    MethodToolHandler,
    ModelAdapter,
    ModelConfig,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelToolCall,
    OpenAIModelAdapter,
    RunStatus,
    SessionRecorder,
    StepOutcome,
    StopReason,
    ToolControl,
    TurnCompleted,
    TurnStarted,
)

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Finish the current task.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
}

TEST_CONFIG = ModelConfig(
    model="test-model",
    api_key="test-key",
    base_url="https://example.invalid",
)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class StreamingTextModel(ModelAdapter):
    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        raise AssertionError("stream() should be used by the orchestrator")

    async def stream(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        yield ModelTextDelta("Hel")
        yield ModelTextDelta("lo")
        yield ModelResponseCompleted(AssistantMessage(content="Hello"))


class CompleteOnlyModel(ModelAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        self.calls += 1
        return AssistantMessage(content="fallback")


class StreamingThinkingModel(ModelAdapter):
    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        raise AssertionError("stream() should be used by the orchestrator")

    async def stream(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        yield ModelThinkingDelta("inspect ")
        yield ModelThinkingDelta("context")
        yield ModelTextDelta("final")
        yield ModelResponseCompleted(AssistantMessage(content="final"))


class BlockingStreamModel(ModelAdapter):
    def __init__(self) -> None:
        self.blocked = asyncio.Event()
        self.closed = asyncio.Event()

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        raise AssertionError("stream() should be used by the orchestrator")

    async def stream(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        yield ModelTextDelta("partial")
        self.blocked.set()
        try:
            await asyncio.Future()
        finally:
            self.closed.set()


class BlockingThinkingStreamModel(ModelAdapter):
    def __init__(self) -> None:
        self.blocked = asyncio.Event()
        self.closed = asyncio.Event()

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        raise AssertionError("stream() should be used by the orchestrator")

    async def stream(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        yield ModelThinkingDelta("unfinished reasoning")
        self.blocked.set()
        try:
            await asyncio.Future()
        finally:
            self.closed.set()


class ToolStreamModel(ModelAdapter):
    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        raise AssertionError("stream() should be used by the orchestrator")

    async def stream(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        yield ModelResponseCompleted(
            AssistantMessage(
                tool_calls=(
                    ModelToolCall(
                        id="finish-1",
                        name="finish",
                        arguments='{"summary":"streamed tool"}',
                    ),
                )
            )
        )


class MissingTerminalModel(ModelAdapter):
    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        raise AssertionError("stream() should be used by the orchestrator")

    async def stream(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        yield ModelTextDelta("unfinished")


class FinishHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((FINISH_TOOL,))

    async def do_finish(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome(
            {"summary": arguments["summary"]},
            control=ToolControl.COMPLETE,
        )


class FakeOpenAIStream:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> "FakeOpenAIStream":
        return self

    async def __anext__(self) -> Any:
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


class FakeOpenAICompletions:
    def __init__(self, response: FakeOpenAIStream) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeOpenAIStream:
        self.calls.append(kwargs)
        return self.response


def chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
    reasoning: str | None = None,
    reasoning_text: str | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                    reasoning=reasoning,
                    reasoning_text=reasoning_text,
                ),
            )
        ]
    )


def payloads(sink: RecordingSink) -> list[object]:
    return [event.payload for event in sink.events]


class AgentStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_deltas_are_observed_before_atomic_message_commit(
        self,
    ) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="stream", storage=storage)
        observer = RecordingSink()
        agent = BaseAgent(
            StreamingTextModel(),
            agent_id="stream-agent",
            event_sink=CompositeAgentEventSink([observer, recorder]),
        )

        result = await agent.run(task="stream text")
        session = await recorder.load()

        self.assertEqual(result.output, "Hello")
        self.assertEqual(
            [type(payload) for payload in payloads(observer)],
            [
                AgentStarted,
                TurnStarted,
                AssistantTextDelta,
                AssistantTextDelta,
                MessageCompleted,
                TurnCompleted,
                AgentFinished,
            ],
        )
        deltas = [
            payload.delta
            for payload in payloads(observer)
            if isinstance(payload, AssistantTextDelta)
        ]
        self.assertEqual(deltas, ["Hel", "lo"])
        self.assertEqual(agent.messages[-1]["content"], "Hello")
        assert session is not None
        self.assertEqual(
            [message["content"] for message in session.messages],
            ["stream text", "Hello"],
        )
        self.assertEqual(
            [entry.sequence for entry in session.entries],
            [1, 5],
        )

    async def test_complete_only_adapter_uses_default_stream_fallback(
        self,
    ) -> None:
        model = CompleteOnlyModel()
        sink = RecordingSink()
        agent = BaseAgent(model, agent_id="fallback", event_sink=sink)

        result = await agent.run(task="fallback")

        self.assertEqual(result.output, "fallback")
        self.assertEqual(model.calls, 1)
        self.assertFalse(
            any(isinstance(payload, AssistantTextDelta) for payload in payloads(sink))
        )

    async def test_thinking_deltas_are_observable_but_not_persisted(
        self,
    ) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="thinking", storage=storage)
        observer = RecordingSink()
        agent = BaseAgent(
            StreamingThinkingModel(),
            agent_id="thinking-agent",
            event_sink=CompositeAgentEventSink([observer, recorder]),
        )

        result = await agent.run(task="reason then answer")
        session = await recorder.load()

        self.assertEqual(result.output, "final")
        self.assertEqual(
            [type(payload) for payload in payloads(observer)],
            [
                AgentStarted,
                TurnStarted,
                AssistantThinkingDelta,
                AssistantThinkingDelta,
                AssistantTextDelta,
                MessageCompleted,
                TurnCompleted,
                AgentFinished,
            ],
        )
        thinking = [
            payload.delta
            for payload in payloads(observer)
            if isinstance(payload, AssistantThinkingDelta)
        ]
        self.assertEqual(thinking, ["inspect ", "context"])
        self.assertNotIn("thinking", agent.messages[-1])
        assert session is not None
        self.assertEqual(
            [message["content"] for message in session.messages],
            ["reason then answer", "final"],
        )

    async def test_abort_discards_partial_message_but_finishes_session(
        self,
    ) -> None:
        model = BlockingStreamModel()
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="abort-stream", storage=storage)
        observer = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="abort-stream",
            event_sink=CompositeAgentEventSink([observer, recorder]),
        )
        run = asyncio.create_task(agent.run(task="abort partial"))
        await model.blocked.wait()

        agent.abort("stop streaming")
        result = await run
        session = await recorder.load()

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertTrue(model.closed.is_set())
        self.assertFalse(
            any(isinstance(payload, MessageCompleted) for payload in payloads(observer))
        )
        self.assertIsInstance(payloads(observer)[-2], TurnCompleted)
        self.assertIsInstance(payloads(observer)[-1], AgentFinished)
        self.assertEqual(
            [message["role"] for message in agent.messages[-1:]],
            ["user"],
        )
        assert session is not None
        self.assertEqual(
            [message["role"] for message in session.messages],
            ["user"],
        )
        self.assertTrue(session.runs[0].finished)

    async def test_streamed_final_tool_call_executes_normally(self) -> None:
        agent = BaseAgent(
            ToolStreamModel(),
            agent_id="stream-tool",
            handlers=[FinishHandler()],
        )

        result = await agent.run(task="finish with a tool")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.stop_reason, StopReason.TOOL_COMPLETION)
        self.assertIn("streamed tool", result.output or "")

    async def test_abort_during_thinking_discards_provisional_reasoning(
        self,
    ) -> None:
        model = BlockingThinkingStreamModel()
        storage = MemorySessionStorage()
        recorder = SessionRecorder(
            session_id="abort-thinking",
            storage=storage,
        )
        observer = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="abort-thinking",
            event_sink=CompositeAgentEventSink([observer, recorder]),
        )
        run = asyncio.create_task(agent.run(task="abort reasoning"))
        await model.blocked.wait()

        agent.abort("stop thinking")
        result = await run
        session = await recorder.load()

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertTrue(model.closed.is_set())
        self.assertTrue(
            any(
                isinstance(payload, AssistantThinkingDelta)
                for payload in payloads(observer)
            )
        )
        self.assertFalse(
            any(isinstance(payload, MessageCompleted) for payload in payloads(observer))
        )
        assert session is not None
        self.assertEqual(
            [message["role"] for message in session.messages],
            ["user"],
        )

    async def test_missing_stream_terminal_event_is_runtime_failure(self) -> None:
        agent = BaseAgent(MissingTerminalModel(), agent_id="missing-terminal")

        result = await agent.run(task="malformed stream")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.stop_reason, StopReason.RUNTIME_ERROR)
        self.assertIn("without a completed response", result.error or "")
        self.assertEqual(agent.messages[-1]["role"], "user")

    async def test_openai_stream_normalizes_text_and_tool_call_chunks(
        self,
    ) -> None:
        first_tool_delta = SimpleNamespace(
            index=0,
            id="call-1",
            function=SimpleNamespace(name="fin", arguments='{"sum'),
        )
        second_tool_delta = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(
                name="ish",
                arguments='mary":"ok"}',
            ),
        )
        response = FakeOpenAIStream(
            [
                chunk(
                    reasoning_content="reason ",
                    reasoning="reason ",
                ),
                chunk(reasoning_text="carefully"),
                chunk(content="Hello "),
                chunk(content="world"),
                chunk(tool_calls=[first_tool_delta]),
                chunk(tool_calls=[second_tool_delta]),
                chunk(finish_reason="tool_calls"),
                SimpleNamespace(choices=[]),
            ]
        )
        completions = FakeOpenAICompletions(response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = OpenAIModelAdapter(
            TEST_CONFIG,
            client=client,  # type: ignore[arg-type]
        )
        context = AgentContextBuilder().build(AgentState())

        events = [event async for event in adapter.stream(context)]

        self.assertEqual(
            [event.delta for event in events if isinstance(event, ModelTextDelta)],
            ["Hello ", "world"],
        )
        self.assertEqual(
            [event.delta for event in events if isinstance(event, ModelThinkingDelta)],
            ["reason ", "carefully"],
        )
        terminal = events[-1]
        assert isinstance(terminal, ModelResponseCompleted)
        self.assertEqual(terminal.message.content, "Hello world")
        self.assertEqual(terminal.message.tool_calls[0].id, "call-1")
        self.assertEqual(terminal.message.tool_calls[0].name, "finish")
        self.assertEqual(
            terminal.message.tool_calls[0].arguments,
            '{"summary":"ok"}',
        )
        self.assertTrue(completions.calls[0]["stream"])
        self.assertTrue(response.closed)

    async def test_openai_stream_rejects_silent_incomplete_response(
        self,
    ) -> None:
        response = FakeOpenAIStream([chunk(content="partial")])
        completions = FakeOpenAICompletions(response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = OpenAIModelAdapter(
            TEST_CONFIG,
            client=client,  # type: ignore[arg-type]
        )
        context = AgentContextBuilder().build(AgentState())

        with self.assertRaisesRegex(RuntimeError, "without finish_reason"):
            _ = [event async for event in adapter.stream(context)]

        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
