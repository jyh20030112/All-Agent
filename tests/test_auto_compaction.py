import asyncio
import unittest
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ejagent import (
    AgentEvent,
    AssistantMessage,
    AssistantTextDelta,
    AutoCompactionPolicy,
    BaseAgent,
    CancellationToken,
    CompactionCompleted,
    CompactionFailed,
    CompactionPolicy,
    CompactionRequest,
    CompactionStarted,
    CompactionTrigger,
    CompactorOutput,
    ContextBudget,
    ContextOverflowError,
    ContextPressureEvaluated,
    MemorySessionStorage,
    ModelAdapter,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    RunStatus,
    SessionRecorder,
    SteeringApplied,
    StopReason,
)


class MarkerTokenEstimator:
    def estimate_message(self, message: Mapping[str, Any]) -> int:
        return int(message.get("_tokens", 0))

    def estimate_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        return 0


class ScriptedStreamModel(ModelAdapter):
    def __init__(
        self,
        outcomes: list[Exception | list[ModelStreamEvent | Exception]],
    ) -> None:
        self.outcomes = list(outcomes)
        self.contexts: list[Any] = []

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
        self.contexts.append(context)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        for item in outcome:
            if isinstance(item, Exception):
                raise item
            yield item


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class StaticCompactor:
    def __init__(self) -> None:
        self.requests: list[CompactionRequest] = []

    async def compact(
        self,
        request: CompactionRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> CompactorOutput:
        self.requests.append(request)
        return CompactorOutput("summary of old turns", "test-compactor")


class FailingCompactor:
    async def compact(
        self,
        request: CompactionRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> CompactorOutput:
        raise RuntimeError("summary provider failed")


class BlockingCompactor:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def compact(
        self,
        request: CompactionRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> CompactorOutput:
        self.started.set()
        await asyncio.Event().wait()
        return CompactorOutput("unreachable", "test-compactor")


class ReleasableCompactor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def compact(
        self,
        request: CompactionRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> CompactorOutput:
        self.started.set()
        await self.release.wait()
        return CompactorOutput("summary before steering", "test-compactor")


def compaction_policy() -> CompactionPolicy:
    return CompactionPolicy(
        ContextBudget(
            context_window=100,
            reserve_tokens=10,
            keep_recent_tokens=20,
        )
    )


def history() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "old task", "_tokens": 60},
        {"role": "assistant", "content": "old answer", "_tokens": 60},
        {"role": "user", "content": "recent task", "_tokens": 10},
        {"role": "assistant", "content": "recent answer", "_tokens": 10},
    ]


def completed(text: str = "done") -> list[ModelStreamEvent | Exception]:
    return [ModelResponseCompleted(AssistantMessage(content=text))]


class AutoCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_automatic_behavior_is_off_by_default(self) -> None:
        model = ScriptedStreamModel([completed()])
        compactor = StaticCompactor()
        agent = BaseAgent(
            model,
            agent_id="auto-default-off",
            compaction_policy=compaction_policy(),
            compactor=compactor,
            context_token_estimator=MarkerTokenEstimator(),
        )
        agent.reset(history=history())

        result = await agent.run(task="continue")

        self.assertTrue(result.succeeded)
        self.assertEqual(compactor.requests, [])
        self.assertIn("old task", repr(model.contexts[0].agent_messages))

    async def test_pressure_compacts_and_rebuilds_context_in_same_run(self) -> None:
        model = ScriptedStreamModel([completed()])
        compactor = StaticCompactor()
        sink = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="auto-pressure",
            compaction_policy=compaction_policy(),
            compactor=compactor,
            auto_compaction_policy=AutoCompactionPolicy(),
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(history=history())

        result = await agent.run(task="continue")

        self.assertTrue(result.succeeded)
        self.assertEqual(len(compactor.requests), 1)
        self.assertEqual(compactor.requests[0].trigger, CompactionTrigger.PRESSURE)
        self.assertNotIn("old task", repr(model.contexts[0].agent_messages))
        self.assertIn("Conversation summary", repr(model.contexts[0].llm_messages))
        relevant = [
            event
            for event in sink.events
            if isinstance(
                event.payload,
                (ContextPressureEvaluated, CompactionStarted, CompactionCompleted),
            )
        ]
        self.assertEqual(
            [type(event.payload) for event in relevant],
            [
                ContextPressureEvaluated,
                CompactionStarted,
                CompactionCompleted,
                ContextPressureEvaluated,
            ],
        )
        started = relevant[1].payload
        finished = relevant[2].payload
        assert isinstance(started, CompactionStarted)
        assert isinstance(finished, CompactionCompleted)
        self.assertEqual(
            started.request.operation_id,
            finished.result.operation_id,
        )
        self.assertEqual(len({event.run_id for event in sink.events}), 1)
        self.assertEqual(
            [event.sequence for event in sink.events],
            list(range(1, len(sink.events) + 1)),
        )

    async def test_pressure_compactor_failure_stops_safely(self) -> None:
        model = ScriptedStreamModel([completed()])
        agent = BaseAgent(
            model,
            agent_id="auto-failure",
            compaction_policy=compaction_policy(),
            compactor=FailingCompactor(),
            auto_compaction_policy=AutoCompactionPolicy(),
            context_token_estimator=MarkerTokenEstimator(),
        )
        agent.reset(history=history())

        result = await agent.run(task="continue")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.stop_reason, StopReason.COMPACTION_FAILED)
        self.assertEqual(result.error, "summary provider failed")
        self.assertEqual(model.contexts, [])
        self.assertIn("old task", repr(agent.messages))
        self.assertNotIn("_ejagent_summary", repr(agent.messages))

    async def test_abort_cancels_automatic_compaction(self) -> None:
        model = ScriptedStreamModel([completed()])
        compactor = BlockingCompactor()
        sink = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="auto-abort",
            compaction_policy=compaction_policy(),
            compactor=compactor,
            auto_compaction_policy=AutoCompactionPolicy(),
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(history=history())

        task = asyncio.create_task(agent.run(task="continue"))
        await compactor.started.wait()
        self.assertTrue(agent.abort("stop automatic compaction"))
        result = await task

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertEqual(model.contexts, [])
        failures = [
            event.payload
            for event in sink.events
            if isinstance(event.payload, CompactionFailed)
        ]
        self.assertEqual(len(failures), 1)

    async def test_context_overflow_compacts_and_retries_once(self) -> None:
        model = ScriptedStreamModel(
            [ContextOverflowError("too large"), completed("recovered")]
        )
        compactor = StaticCompactor()
        agent = BaseAgent(
            model,
            agent_id="overflow-recovery",
            compaction_policy=compaction_policy(),
            compactor=compactor,
            auto_compaction_policy=AutoCompactionPolicy(compact_on_pressure=False),
            context_token_estimator=MarkerTokenEstimator(),
        )
        agent.reset(history=history())

        result = await agent.run(task="continue")

        self.assertTrue(result.succeeded)
        self.assertEqual(result.output, "recovered")
        self.assertEqual(len(model.contexts), 2)
        self.assertEqual(len(compactor.requests), 1)
        self.assertEqual(compactor.requests[0].trigger, CompactionTrigger.OVERFLOW)
        self.assertIn("old task", repr(model.contexts[0].agent_messages))
        self.assertNotIn("old task", repr(model.contexts[1].agent_messages))
        self.assertEqual(result.usage.request_count, 2)

    async def test_steering_during_overflow_compaction_reaches_retry(self) -> None:
        model = ScriptedStreamModel(
            [ContextOverflowError("too large"), completed("steered retry")]
        )
        compactor = ReleasableCompactor()
        sink = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="overflow-steering",
            compaction_policy=compaction_policy(),
            compactor=compactor,
            auto_compaction_policy=AutoCompactionPolicy(compact_on_pressure=False),
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(history=history())
        run = asyncio.create_task(agent.run(task="continue"))
        await compactor.started.wait()

        receipt = await agent.steer("use the corrected direction")
        compactor.release.set()
        result = await run

        self.assertTrue(receipt.accepted)
        self.assertEqual(result.output, "steered retry")
        self.assertNotIn(
            "use the corrected direction",
            repr(model.contexts[0].agent_messages),
        )
        self.assertIn(
            "use the corrected direction",
            repr(model.contexts[1].agent_messages),
        )
        applied = [
            event.payload
            for event in sink.events
            if isinstance(event.payload, SteeringApplied)
        ]
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0].target_turn, 1)

    async def test_second_context_overflow_is_structured_failure(self) -> None:
        model = ScriptedStreamModel(
            [ContextOverflowError("first"), ContextOverflowError("second")]
        )
        compactor = StaticCompactor()
        agent = BaseAgent(
            model,
            agent_id="overflow-limit",
            compaction_policy=compaction_policy(),
            compactor=compactor,
            auto_compaction_policy=AutoCompactionPolicy(compact_on_pressure=False),
            context_token_estimator=MarkerTokenEstimator(),
        )
        agent.reset(history=history())

        result = await agent.run(task="continue")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.stop_reason, StopReason.CONTEXT_OVERFLOW)
        self.assertEqual(len(model.contexts), 2)
        self.assertEqual(len(compactor.requests), 1)

    async def test_overflow_after_delta_is_not_retried(self) -> None:
        model = ScriptedStreamModel(
            [
                [
                    ModelTextDelta("partial"),
                    ContextOverflowError("late overflow"),
                ]
            ]
        )
        compactor = StaticCompactor()
        sink = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="overflow-after-delta",
            compaction_policy=compaction_policy(),
            compactor=compactor,
            auto_compaction_policy=AutoCompactionPolicy(compact_on_pressure=False),
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(history=history())

        result = await agent.run(task="continue")

        self.assertEqual(result.stop_reason, StopReason.CONTEXT_OVERFLOW)
        self.assertEqual(len(model.contexts), 1)
        self.assertEqual(compactor.requests, [])
        self.assertEqual(
            len(
                [
                    event
                    for event in sink.events
                    if isinstance(event.payload, AssistantTextDelta)
                ]
            ),
            1,
        )

    async def test_session_records_automatic_compaction_operation(self) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="auto-session", storage=storage)
        model = ScriptedStreamModel([completed()])
        agent = BaseAgent(
            model,
            agent_id="auto-session-agent",
            compaction_policy=compaction_policy(),
            compactor=StaticCompactor(),
            auto_compaction_policy=AutoCompactionPolicy(),
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=recorder,
        )
        agent.reset(history=history())

        result = await agent.run(task="continue")
        session = await recorder.load()

        self.assertTrue(result.succeeded)
        assert session is not None
        self.assertEqual(len(session.compactions), 1)
        self.assertTrue(session.compactions[0].operation_id)
        self.assertNotEqual(
            session.compactions[0].operation_id,
            session.runs[0].run_id,
        )

    async def test_enabled_policy_requires_compaction_dependencies(self) -> None:
        model = ScriptedStreamModel([completed()])
        with self.assertRaisesRegex(ValueError, "CompactionPolicy"):
            BaseAgent(
                model,
                agent_id="auto-missing-policy",
                compactor=StaticCompactor(),
                auto_compaction_policy=AutoCompactionPolicy(),
            )
        with self.assertRaisesRegex(ValueError, "Compactor"):
            BaseAgent(
                model,
                agent_id="auto-missing-compactor",
                compaction_policy=compaction_policy(),
                auto_compaction_policy=AutoCompactionPolicy(),
            )


if __name__ == "__main__":
    unittest.main()
