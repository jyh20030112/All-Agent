import asyncio
import unittest
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from ejagent import (
    AgentEvent,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    CompactionCompleted,
    CompactionFailed,
    CompactionPolicy,
    CompactionRequest,
    CompactionStarted,
    CompactionStatus,
    CompactorOutput,
    ContextBudget,
    MemorySessionStorage,
    ModelAdapter,
    SessionRecorder,
    SummaryEntry,
)


class MarkerTokenEstimator:
    def estimate_message(self, message: Mapping[str, Any]) -> int:
        return int(message.get("_tokens", 0))

    def estimate_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        return 0


class ConstantTokenEstimator:
    def estimate_message(self, message: Mapping[str, Any]) -> int:
        return 100

    def estimate_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        return 0


class QueueModel(ModelAdapter):
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or ["done"])
        self.contexts: list[Any] = []

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        self.contexts.append(context)
        return AssistantMessage(content=self.responses.pop(0))


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class StaticCompactor:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or ["old work was summarized"])
        self.requests: list[CompactionRequest] = []

    async def compact(
        self,
        request: CompactionRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> CompactorOutput:
        self.requests.append(request)
        return CompactorOutput(
            content=self.outputs.pop(0),
            source="static-test-compactor",
        )


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
        self.stopped = asyncio.Event()

    async def compact(
        self,
        request: CompactionRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> CompactorOutput:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()
        return CompactorOutput("unreachable", "blocking")


def policy() -> CompactionPolicy:
    return CompactionPolicy(
        ContextBudget(
            context_window=1000,
            reserve_tokens=100,
            keep_recent_tokens=50,
        )
    )


def history() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "old task", "_tokens": 100},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "old-call"}],
            "_tokens": 10,
        },
        {
            "role": "tool",
            "tool_call_id": "old-call",
            "content": "large old output",
            "_tokens": 100,
        },
        {"role": "assistant", "content": "old answer", "_tokens": 10},
        {"role": "user", "content": "recent task", "_tokens": 30},
        {"role": "assistant", "content": "recent answer", "_tokens": 30},
    ]


class ContextCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_compaction_requires_policy_and_compactor(self) -> None:
        agent = BaseAgent(QueueModel(), agent_id="compact-config")

        with self.assertRaisesRegex(RuntimeError, "requires a Compactor"):
            await agent.compact()
        with self.assertRaisesRegex(RuntimeError, "requires a CompactionPolicy"):
            await agent.compact(compactor=StaticCompactor())
        await agent.wait_for_idle()

    async def test_explicit_compaction_atomically_replaces_old_turns(self) -> None:
        model = QueueModel()
        compactor = StaticCompactor()
        sink = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="compact-success",
            compaction_policy=policy(),
            compactor=compactor,
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(history=history())

        result = await agent.compact()

        self.assertEqual(result.status, CompactionStatus.COMPLETED)
        self.assertTrue(result.completed)
        self.assertIsInstance(result.summary, SummaryEntry)
        assert result.summary is not None
        self.assertEqual(result.summary.summarized_message_count, 4)
        self.assertEqual(result.summary.tokens_before, 280)
        self.assertEqual(
            [message["role"] for message in agent.messages],
            ["system", "system", "user", "assistant"],
        )
        summary_message = agent.messages[1]
        self.assertIn("_ejagent_summary", summary_message)
        self.assertNotIn("old-call", repr(agent.messages))
        self.assertEqual(
            [type(event.payload) for event in sink.events],
            [CompactionStarted, CompactionCompleted],
        )
        self.assertEqual([event.sequence for event in sink.events], [1, 2])
        self.assertEqual(len({event.run_id for event in sink.events}), 1)
        self.assertEqual(
            [message["role"] for message in result.messages],
            ["system", "user", "assistant"],
        )

        await agent.run(task="continue from the summary")
        sent_summary = model.contexts[0].llm_messages[1]
        self.assertNotIn("_ejagent_summary", sent_summary)
        self.assertIn("Conversation summary", sent_summary["content"])

    async def test_compactor_failure_keeps_history_unchanged(self) -> None:
        sink = RecordingSink()
        agent = BaseAgent(
            QueueModel(),
            agent_id="compact-failure",
            compaction_policy=policy(),
            compactor=FailingCompactor(),
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(history=history())
        before = deepcopy(agent.messages)

        result = await agent.compact()

        self.assertEqual(result.status, CompactionStatus.FAILED)
        self.assertEqual(result.error, "summary provider failed")
        self.assertEqual(agent.messages, before)
        self.assertEqual(
            [type(event.payload) for event in sink.events],
            [CompactionStarted, CompactionFailed],
        )

    async def test_abort_cancels_compactor_and_preserves_history(self) -> None:
        compactor = BlockingCompactor()
        sink = RecordingSink()
        agent = BaseAgent(
            QueueModel(),
            agent_id="compact-abort",
            compaction_policy=policy(),
            compactor=compactor,
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(history=history())
        before = deepcopy(agent.messages)

        task = asyncio.create_task(agent.compact())
        await compactor.started.wait()
        self.assertTrue(agent.abort("stop compaction"))
        result = await task
        await agent.wait_for_idle()

        self.assertEqual(result.status, CompactionStatus.CANCELLED)
        self.assertEqual(result.error, "stop compaction")
        self.assertEqual(agent.messages, before)
        self.assertTrue(compactor.stopped.is_set())
        self.assertEqual(
            [type(event.payload) for event in sink.events],
            [CompactionStarted, CompactionFailed],
        )

    async def test_caller_cancellation_emits_failure_and_preserves_history(
        self,
    ) -> None:
        compactor = BlockingCompactor()
        sink = RecordingSink()
        agent = BaseAgent(
            QueueModel(),
            agent_id="compact-caller-cancel",
            compaction_policy=policy(),
            compactor=compactor,
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(history=history())
        before = deepcopy(agent.messages)

        task = asyncio.create_task(agent.compact())
        await compactor.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await agent.wait_for_idle()

        self.assertEqual(agent.messages, before)
        self.assertTrue(compactor.stopped.is_set())
        failed = sink.events[-1].payload
        self.assertIsInstance(failed, CompactionFailed)
        assert isinstance(failed, CompactionFailed)
        self.assertEqual(failed.result.status, CompactionStatus.CANCELLED)

    async def test_single_turn_is_safely_skipped(self) -> None:
        compactor = StaticCompactor()
        sink = RecordingSink()
        agent = BaseAgent(
            QueueModel(),
            agent_id="compact-skip",
            compaction_policy=policy(),
            compactor=compactor,
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.reset(
            history=[
                {"role": "user", "content": "only", "_tokens": 100},
                {"role": "assistant", "content": "turn", "_tokens": 100},
            ]
        )
        before = deepcopy(agent.messages)

        result = await agent.compact()

        self.assertEqual(result.status, CompactionStatus.SKIPPED)
        self.assertEqual(agent.messages, before)
        self.assertEqual(compactor.requests, [])
        self.assertEqual(
            [type(event.payload) for event in sink.events],
            [CompactionStarted, CompactionCompleted],
        )

    async def test_repeated_compaction_replaces_and_merges_summary(self) -> None:
        compactor = StaticCompactor(["first summary", "merged summary"])
        agent = BaseAgent(
            QueueModel(),
            agent_id="compact-repeat",
            compaction_policy=policy(),
            compactor=compactor,
            context_token_estimator=MarkerTokenEstimator(),
        )
        agent.reset(history=history())

        first = await agent.compact()
        agent.state.add_messages(
            [
                {"role": "user", "content": "next old", "_tokens": 100},
                {
                    "role": "assistant",
                    "content": "next old answer",
                    "_tokens": 100,
                },
                {"role": "user", "content": "newest", "_tokens": 30},
                {
                    "role": "assistant",
                    "content": "newest answer",
                    "_tokens": 30,
                },
            ]
        )
        second = await agent.compact()

        self.assertEqual(first.status, CompactionStatus.COMPLETED)
        self.assertEqual(second.status, CompactionStatus.COMPLETED)
        self.assertIsNotNone(compactor.requests[1].previous_summary)
        self.assertEqual(
            compactor.requests[1].previous_summary.content,
            "first summary",
        )
        summary_messages = [
            message for message in agent.messages if "_ejagent_summary" in message
        ]
        self.assertEqual(len(summary_messages), 1)
        assert second.summary is not None
        self.assertGreater(second.summary.summarized_message_count, 4)
        self.assertEqual(second.summary.content, "merged summary")

    async def test_session_projection_preserves_late_context_barrier(self) -> None:
        agent = BaseAgent(
            QueueModel(),
            agent_id="compact-barrier",
            compaction_policy=policy(),
            compactor=StaticCompactor(),
            context_token_estimator=MarkerTokenEstimator(),
        )
        agent.reset(
            history=[
                {"role": "user", "content": "protected old", "_tokens": 100},
                {
                    "role": "assistant",
                    "content": "protected answer",
                    "_tokens": 100,
                },
                {"role": "system", "content": "late barrier", "_tokens": 1},
                {"role": "user", "content": "summarize", "_tokens": 100},
                {
                    "role": "assistant",
                    "content": "summarize answer",
                    "_tokens": 100,
                },
                {"role": "user", "content": "recent", "_tokens": 30},
                {
                    "role": "assistant",
                    "content": "recent answer",
                    "_tokens": 30,
                },
            ]
        )

        result = await agent.compact()

        self.assertEqual(result.status, CompactionStatus.COMPLETED)
        self.assertEqual(
            [message.get("content") for message in result.messages[:3]],
            ["protected old", "protected answer", "late barrier"],
        )

    async def test_session_restores_compacted_projection_and_new_runs(
        self,
    ) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(
            session_id="compacted-session",
            storage=storage,
        )
        model = QueueModel(["continued"])
        agent = BaseAgent(
            model,
            agent_id="compact-session-agent",
            compaction_policy=policy(),
            compactor=StaticCompactor(),
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=recorder,
        )
        agent.reset(history=history())

        compacted = await agent.compact()
        await agent.run(task="new task")
        session = await recorder.load()

        self.assertEqual(compacted.status, CompactionStatus.COMPLETED)
        assert session is not None
        self.assertEqual(len(session.compactions), 1)
        self.assertEqual(len(session.entries), 2)
        self.assertEqual(
            [message["role"] for message in session.messages],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertIn("_ejagent_summary", session.messages[0])

        resumed_model = QueueModel(["resumed"])
        resumed = BaseAgent(
            resumed_model,
            agent_id="compact-session-agent",
        )
        resumed.reset(session.messages)
        await resumed.run(task="resume")

        llm_summary = resumed_model.contexts[0].llm_messages[1]
        self.assertIn("Conversation summary", llm_summary["content"])
        self.assertNotIn("_ejagent_summary", llm_summary)

    async def test_session_keeps_audit_entries_covered_by_compaction(
        self,
    ) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(
            session_id="compaction-audit",
            storage=storage,
        )
        agent = BaseAgent(
            QueueModel(["one", "two", "three", "four"]),
            agent_id="compaction-audit-agent",
            compaction_policy=policy(),
            compactor=StaticCompactor(),
            context_token_estimator=ConstantTokenEstimator(),
            event_sink=recorder,
        )

        await agent.run(task="first")
        await agent.run(task="second")
        await agent.run(task="third")
        result = await agent.compact()
        await agent.run(task="fourth")
        session = await recorder.load()

        self.assertEqual(result.status, CompactionStatus.COMPLETED)
        assert session is not None
        self.assertEqual(len(session.entries), 8)
        self.assertEqual(len(session.runs), 4)
        self.assertEqual(len(session.compactions), 1)
        self.assertEqual(session.compactions[0].covered_entry_count, 6)
        self.assertEqual(
            [message["content"] for message in session.messages[1:]],
            ["third", "three", "fourth", "four"],
        )


if __name__ == "__main__":
    unittest.main()
