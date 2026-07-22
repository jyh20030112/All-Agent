import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from ejagent import (
    AgentContextBuilder,
    AgentEvent,
    AgentState,
    AssistantMessage,
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
    ModelToolCall,
    ModelUsage,
    OpenAIModelAdapter,
    RunStatus,
    RuntimePolicy,
    RunUsage,
    SessionRecorder,
    StepOutcome,
    StopReason,
)

CONTINUE_TOOL = {
    "type": "function",
    "function": {
        "name": "continue_work",
        "description": "Record one completed unit of work.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def usage(
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> ModelUsage:
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=cache_read_tokens,
        reasoning_tokens=reasoning_tokens,
    )


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class UsageSequenceModel(ModelAdapter):
    def __init__(
        self,
        responses: list[tuple[AssistantMessage, ModelUsage | None]],
    ) -> None:
        self.responses = list(responses)
        self.contexts: list[Any] = []
        self.calls = 0

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        raise AssertionError("stream() should be used")

    async def stream(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        self.contexts.append(context)
        message, response_usage = self.responses.pop(0)
        yield ModelResponseCompleted(message, response_usage)


class CompleteOnlyModel(ModelAdapter):
    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        return AssistantMessage(content="legacy complete")


class ContinueHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((CONTINUE_TOOL,))
        self.calls = 0

    async def do_continue_work(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        self.calls += 1
        return StepOutcome({"completed": self.calls})


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


def tool_call() -> ModelToolCall:
    return ModelToolCall(
        id="continue-1",
        name="continue_work",
        arguments="{}",
    )


class UsageBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_usage_is_aggregated_persisted_and_removed_from_llm(
        self,
    ) -> None:
        first_usage = usage(
            100,
            20,
            cache_read_tokens=30,
            reasoning_tokens=5,
        )
        second_usage = usage(
            150,
            30,
            cache_read_tokens=40,
        )
        model = UsageSequenceModel(
            [
                (AssistantMessage(tool_calls=(tool_call(),)), first_usage),
                (AssistantMessage(content="done"), second_usage),
            ]
        )
        handler = ContinueHandler()
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="usage", storage=storage)
        observer = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="usage-aggregate",
            handlers=[handler],
            event_sink=CompositeAgentEventSink([recorder, observer]),
        )

        result = await agent.run(task="work")
        session = await recorder.load()

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.usage.input_tokens, 250)
        self.assertEqual(result.usage.output_tokens, 50)
        self.assertEqual(result.usage.total_tokens, 300)
        self.assertEqual(result.usage.request_count, 2)
        self.assertEqual(result.usage.reported_request_count, 2)
        self.assertEqual(result.usage.cache_read_tokens, 70)
        self.assertIsNone(result.usage.reasoning_tokens)
        self.assertTrue(result.usage.complete)

        completed = [
            event.payload
            for event in observer.events
            if isinstance(event.payload, MessageCompleted)
        ]
        self.assertEqual(
            [payload.usage for payload in completed],
            [first_usage, second_usage],
        )

        agent_assistant = next(
            message for message in agent.messages if message["role"] == "assistant"
        )
        self.assertEqual(agent_assistant["usage"], first_usage.to_dict())
        context_assistant_index = next(
            index
            for index, message in enumerate(model.contexts[1].agent_messages)
            if message["role"] == "assistant"
        )
        self.assertIn(
            "usage",
            model.contexts[1].agent_messages[context_assistant_index],
        )
        self.assertNotIn(
            "usage",
            model.contexts[1].llm_messages[context_assistant_index],
        )
        self.assertFalse(
            any("usage" in message for message in model.contexts[1].llm_messages)
        )

        assert session is not None
        session_assistant_index = next(
            index
            for index, message in enumerate(session.messages)
            if message["role"] == "assistant"
        )
        self.assertEqual(
            session.messages[session_assistant_index]["usage"],
            first_usage.to_dict(),
        )
        resumed_state = AgentState(messages=session.messages)
        resumed_context = AgentContextBuilder().build(resumed_state)
        self.assertIn(
            "usage",
            resumed_context.agent_messages[session_assistant_index],
        )
        self.assertNotIn(
            "usage",
            resumed_context.llm_messages[session_assistant_index],
        )
        self.assertEqual(session.runs[0].result, result)

    async def test_budget_stops_before_next_request_after_tools_settle(
        self,
    ) -> None:
        model = UsageSequenceModel(
            [
                (
                    AssistantMessage(tool_calls=(tool_call(),)),
                    usage(80, 20),
                ),
                (AssistantMessage(content="must not run"), usage(10, 5)),
            ]
        )
        handler = ContinueHandler()
        agent = BaseAgent(
            model,
            agent_id="usage-budget",
            handlers=[handler],
            runtime_policy=RuntimePolicy(max_run_tokens=100),
        )

        result = await agent.run(task="work")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(
            result.stop_reason,
            StopReason.TOKEN_BUDGET_EXCEEDED,
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(handler.calls, 1)
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.usage.total_tokens, 100)
        self.assertEqual(
            [message["role"] for message in agent.messages[-3:]],
            ["user", "assistant", "tool"],
        )

    async def test_final_response_can_complete_after_crossing_budget(self) -> None:
        model = UsageSequenceModel([(AssistantMessage(content="done"), usage(100, 50))])
        agent = BaseAgent(
            model,
            agent_id="final-over-budget",
            runtime_policy=RuntimePolicy(max_run_tokens=100),
        )

        result = await agent.run(task="finish")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.stop_reason, StopReason.TEXT_RESPONSE)
        self.assertEqual(result.usage.total_tokens, 150)
        self.assertEqual(model.calls, 1)

    async def test_missing_usage_stops_only_when_another_request_is_needed(
        self,
    ) -> None:
        model = UsageSequenceModel(
            [
                (AssistantMessage(tool_calls=(tool_call(),)), None),
                (AssistantMessage(content="must not run"), usage(10, 5)),
            ]
        )
        handler = ContinueHandler()
        agent = BaseAgent(
            model,
            agent_id="missing-usage-budget",
            handlers=[handler],
            runtime_policy=RuntimePolicy(max_run_tokens=100),
        )

        result = await agent.run(task="work")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.stop_reason, StopReason.USAGE_UNAVAILABLE)
        self.assertEqual(model.calls, 1)
        self.assertEqual(handler.calls, 1)
        self.assertEqual(result.usage.request_count, 1)
        self.assertEqual(result.usage.reported_request_count, 0)
        self.assertFalse(result.usage.complete)

    async def test_complete_only_adapter_remains_compatible_without_budget(
        self,
    ) -> None:
        agent = BaseAgent(CompleteOnlyModel(), agent_id="legacy-usage")

        result = await agent.run(task="finish")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.usage.request_count, 1)
        self.assertEqual(result.usage.reported_request_count, 0)
        self.assertFalse(result.usage.complete)

    async def test_usage_accumulator_is_reset_for_each_run(self) -> None:
        model = UsageSequenceModel(
            [
                (AssistantMessage(content="first"), usage(10, 5)),
                (AssistantMessage(content="second"), usage(20, 5)),
            ]
        )
        agent = BaseAgent(model, agent_id="usage-reset")

        first = await agent.run(task="first")
        second = await agent.run(task="second")

        self.assertEqual(first.usage.total_tokens, 15)
        self.assertEqual(second.usage.total_tokens, 25)
        self.assertEqual(first.usage.request_count, 1)
        self.assertEqual(second.usage.request_count, 1)

    async def test_openai_stream_normalizes_usage_only_terminal_chunk(
        self,
    ) -> None:
        response = FakeOpenAIStream(
            [
                SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(
                                content="done",
                                tool_calls=None,
                                reasoning_content=None,
                                reasoning=None,
                                reasoning_text=None,
                            ),
                        )
                    ],
                ),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=120,
                        completion_tokens=30,
                        prompt_tokens_details=SimpleNamespace(
                            cached_tokens=40,
                            cache_write_tokens=10,
                        ),
                        completion_tokens_details=SimpleNamespace(
                            reasoning_tokens=12,
                        ),
                    ),
                ),
            ]
        )
        completions = FakeOpenAICompletions(response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = OpenAIModelAdapter(
            ModelConfig(
                model="test-model",
                api_key="test-key",
                base_url="https://example.invalid",
            ),
            client=client,  # type: ignore[arg-type]
        )
        context = AgentContextBuilder().build(AgentState())

        events = [event async for event in adapter.stream(context)]

        terminal = events[-1]
        assert isinstance(terminal, ModelResponseCompleted)
        self.assertEqual(
            terminal.usage,
            ModelUsage(
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
                cache_read_tokens=40,
                cache_write_tokens=10,
                reasoning_tokens=12,
            ),
        )
        self.assertEqual(
            completions.calls[0]["stream_options"],
            {"include_usage": True},
        )
        self.assertTrue(response.closed)

    async def test_openai_usage_collection_can_be_disabled(self) -> None:
        response = FakeOpenAIStream(
            [
                SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(
                                content="done",
                                tool_calls=None,
                                reasoning_content=None,
                                reasoning=None,
                                reasoning_text=None,
                            ),
                        )
                    ],
                )
            ]
        )
        completions = FakeOpenAICompletions(response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = OpenAIModelAdapter(
            ModelConfig(
                model="test-model",
                api_key="test-key",
                base_url="https://example.invalid",
                include_usage=False,
            ),
            client=client,  # type: ignore[arg-type]
        )

        _ = [
            event
            async for event in adapter.stream(AgentContextBuilder().build(AgentState()))
        ]

        self.assertNotIn("stream_options", completions.calls[0])

    async def test_usage_and_budget_values_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            ModelUsage(10, 5, 14)
        with self.assertRaises(ValueError):
            ModelUsage(10, 5, 15, reasoning_tokens=6)
        with self.assertRaises(ValueError):
            RuntimePolicy(max_run_tokens=0)
        with self.assertRaises(ValueError):
            RunUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                request_count=1,
                reported_request_count=1,
                reasoning_tokens=2,
            )


if __name__ == "__main__":
    unittest.main()
