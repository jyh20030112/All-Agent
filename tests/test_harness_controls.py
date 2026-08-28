from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence

from ejagent.contracts import (
    AssistantMessage,
    CancellationToken,
    ControlStatus,
    FailureCode,
    ModelCallError,
    ModelPort,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    RunAudit,
    RunStatus,
    SessionCommit,
    SessionSnapshot,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutor,
    TransientInstruction,
    UserMessage,
)
from ejagent.harness import (
    AgentHarness,
    FollowUpDiscardedError,
    FollowUpRejectedError,
)


class ScriptedModel(ModelPort):
    def __init__(
        self,
        responses: Sequence[AssistantMessage | ModelCallError],
        *,
        block_first: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []
        self.block_first = block_first
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if self.block_first and len(self.requests) == 1:
            self.started.set()
            await self.release.wait()
        response = self.responses.pop(0)
        if isinstance(response, ModelCallError):
            raise response
        yield ModelResponseCompleted(response)


class NoTools(ToolExecutor):
    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return ()

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        raise AssertionError("no tools are registered")


class BlockingTools(ToolExecutor):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="wait",
            ),
        )

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        self.started.set()
        await self.release.wait()
        return ToolExecutionResult({"released": True})


class BlockingObserver:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.audits: list[RunAudit] = []

    async def observe(self, audit: RunAudit) -> None:
        self.audits.append(audit)
        self.started.set()
        await self.release.wait()


class FailingObserver:
    def __init__(self) -> None:
        self.called = asyncio.Event()

    async def observe(self, audit: RunAudit) -> None:
        self.called.set()
        raise RuntimeError("observer unavailable")


class RecordingStore:
    def __init__(self) -> None:
        self.snapshot: SessionSnapshot | None = None

    async def load(self, agent_id: str) -> SessionSnapshot | None:
        return self.snapshot

    async def commit(self, commit: SessionCommit) -> SessionSnapshot:
        current = self.snapshot
        self.snapshot = SessionSnapshot(
            agent_id=commit.agent_id,
            conversation=commit.resulting_conversation,
            last_result=(
                commit.outcome.result
                if commit.advances_revision
                else current.last_result
                if current is not None
                else None
            ),
        )
        return self.snapshot


class StoreCheckingObserver:
    def __init__(self, store: RecordingStore) -> None:
        self.store = store
        self.called = asyncio.Event()
        self.persisted_revision: int | None = None

    async def observe(self, audit: RunAudit) -> None:
        snapshot = await self.store.load("observed-agent")
        self.persisted_revision = snapshot.revision if snapshot is not None else None
        self.called.set()


class BlockingCommitStore(RecordingStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def commit(self, commit: SessionCommit) -> SessionSnapshot:
        self.started.set()
        await self.release.wait()
        return await super().commit(commit)


class LifecycleObserver:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.called = asyncio.Event()

    async def start(self) -> None:
        self.events.append("observer:start")

    async def observe(self, audit: RunAudit) -> None:
        self.events.append("observer:observe")
        self.called.set()

    async def shutdown(self) -> None:
        self.events.append("observer:shutdown")


class HarnessControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_steering_reaches_next_model_call_without_entering_history(
        self,
    ) -> None:
        call = ToolCall(id="wait-1", name="wait")
        model = ScriptedModel(
            [
                AssistantMessage(tool_calls=(call,)),
                AssistantMessage(content="steered result"),
            ]
        )
        tools = BlockingTools()
        harness = AgentHarness(
            agent_id="steered-agent",
            model=model,
            tools=tools,
            run_id_factory=lambda: "steered-run",
        )
        running = asyncio.create_task(harness.run("original task"))
        await tools.started.wait()

        receipt = harness.steer("use the safer approach")
        tools.release.set()
        outcome = await running

        self.assertTrue(receipt.accepted)
        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(
            model.requests[1].messages[-1],
            TransientInstruction("use the safer approach", "steering"),
        )
        self.assertFalse(
            any(
                isinstance(message, TransientInstruction)
                for message in harness.messages
            )
        )
        applied = [
            record
            for record in outcome.audit_records
            if record.kind == "steering_applied"
        ]
        self.assertEqual(applied[0].payload["input_id"], receipt.input_id)
        self.assertEqual(applied[0].payload["turn"], 2)

    async def test_steering_without_another_safe_point_is_audited_as_discarded(
        self,
    ) -> None:
        model = ScriptedModel(
            [AssistantMessage(content="terminal")],
            block_first=True,
        )
        harness = AgentHarness(
            agent_id="discarded-steering",
            model=model,
            tools=NoTools(),
            run_id_factory=lambda: "discarded-run",
        )
        running = asyncio.create_task(harness.run("task"))
        await model.started.wait()

        receipt = harness.steer("arrived after the safe point")
        model.release.set()
        outcome = await running

        self.assertTrue(receipt.accepted)
        discarded = [
            record
            for record in outcome.audit_records
            if record.kind == "steering_discarded"
        ]
        self.assertEqual(discarded[0].payload["input_id"], receipt.input_id)
        self.assertEqual(discarded[0].payload["reason"], "run_finished")

    async def test_steering_admission_reports_idle_full_and_closed(self) -> None:
        model = ScriptedModel(
            [AssistantMessage(content="done")],
            block_first=True,
        )
        harness = AgentHarness(
            agent_id="steering-capacity",
            model=model,
            tools=NoTools(),
            steering_capacity=1,
        )
        self.assertEqual(harness.steer("idle").status, ControlStatus.NOT_RUNNING)
        running = asyncio.create_task(harness.run("task"))
        await model.started.wait()

        self.assertEqual(harness.steer("accepted").status, ControlStatus.ACCEPTED)
        self.assertEqual(harness.steer("full").status, ControlStatus.QUEUE_FULL)
        model.release.set()
        await running
        await harness.shutdown()

        self.assertEqual(harness.steer("closed").status, ControlStatus.CLOSED)

    async def test_steering_after_last_safe_point_is_rejected_as_too_late(
        self,
    ) -> None:
        store = BlockingCommitStore()
        harness = AgentHarness(
            agent_id="too-late-steering",
            model=ScriptedModel([AssistantMessage(content="done")]),
            tools=NoTools(),
            store=store,
        )
        running = asyncio.create_task(harness.run("task"))
        await store.started.wait()

        receipt = harness.steer("cannot reach a model call")

        self.assertEqual(receipt.status, ControlStatus.TOO_LATE)
        store.release.set()
        await running

    async def test_follow_ups_run_as_independent_fifo_revisions(self) -> None:
        model = ScriptedModel(
            [
                AssistantMessage(content="initial result"),
                AssistantMessage(content="first result"),
                AssistantMessage(content="second result"),
            ],
            block_first=True,
        )
        run_ids = iter(("initial-run", "first-run", "second-run"))
        harness = AgentHarness(
            agent_id="follow-up-agent",
            model=model,
            tools=NoTools(),
            run_id_factory=lambda: next(run_ids),
        )
        running = asyncio.create_task(harness.run("initial"))
        await model.started.wait()

        first = harness.follow_up("first")
        second = harness.follow_up("second")
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(harness.pending_follow_up_count, 2)
        model.release.set()

        initial = await running
        first_outcome = await first.wait()
        second_outcome = await second.wait()

        self.assertEqual(initial.delta.base_revision, 0)
        self.assertEqual(first_outcome.delta.base_revision, 1)
        self.assertEqual(second_outcome.delta.base_revision, 2)
        self.assertEqual(harness.revision, 3)
        user_tasks = [
            message.content
            for message in harness.messages
            if isinstance(message, UserMessage)
        ]
        self.assertEqual(user_tasks, ["initial", "first", "second"])
        self.assertEqual(harness.pending_follow_up_count, 0)

    async def test_follow_up_capacity_and_idle_rejection_are_explicit(self) -> None:
        model = ScriptedModel(
            [
                AssistantMessage(content="initial"),
                AssistantMessage(content="followed"),
            ],
            block_first=True,
        )
        harness = AgentHarness(
            agent_id="follow-up-capacity",
            model=model,
            tools=NoTools(),
            follow_up_capacity=1,
        )
        idle = harness.follow_up("idle")
        with self.assertRaises(FollowUpRejectedError):
            await idle.wait()

        running = asyncio.create_task(harness.run("initial"))
        await model.started.wait()
        accepted = harness.follow_up("accepted")
        full = harness.follow_up("full")

        self.assertEqual(accepted.receipt.status, ControlStatus.ACCEPTED)
        self.assertEqual(full.receipt.status, ControlStatus.QUEUE_FULL)
        with self.assertRaises(FollowUpRejectedError):
            await full.wait()
        model.release.set()
        await running
        await accepted.wait()

    async def test_follow_up_runs_even_when_originating_run_fails(self) -> None:
        model = ScriptedModel(
            [
                ModelCallError(FailureCode.RATE_LIMIT, "busy", retryable=True),
                AssistantMessage(content="recovered"),
            ],
            block_first=True,
        )
        harness = AgentHarness(
            agent_id="follow-up-after-failure",
            model=model,
            tools=NoTools(),
        )
        running = asyncio.create_task(harness.run("initial"))
        await model.started.wait()
        follow_up = harness.follow_up("retry independently")
        model.release.set()

        initial = await running
        followed = await follow_up.wait()

        self.assertEqual(initial.result.status, RunStatus.FAILED)
        self.assertEqual(followed.result.status, RunStatus.COMPLETED)
        self.assertEqual(followed.delta.base_revision, 0)
        self.assertEqual(harness.revision, 1)

    async def test_shutdown_discards_follow_up_that_has_not_started(self) -> None:
        model = ScriptedModel(
            [AssistantMessage(content="unreachable")],
            block_first=True,
        )
        harness = AgentHarness(
            agent_id="follow-up-shutdown",
            model=model,
            tools=NoTools(),
        )
        running = asyncio.create_task(harness.run("blocking"))
        await model.started.wait()
        follow_up = harness.follow_up("discard me")

        await harness.shutdown()
        outcome = await running

        self.assertEqual(outcome.result.status, RunStatus.CANCELLED)
        with self.assertRaises(FollowUpDiscardedError):
            await follow_up.wait()

    async def test_observer_is_not_on_run_critical_path(self) -> None:
        observer = BlockingObserver()
        harness = AgentHarness(
            agent_id="observer-agent",
            model=ScriptedModel([AssistantMessage(content="done")]),
            tools=NoTools(),
            observers=(observer,),
        )

        outcome = await asyncio.wait_for(harness.run("task"), timeout=0.2)
        await observer.started.wait()

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertFalse(observer.release.is_set())
        observer.release.set()
        await harness.shutdown()
        self.assertEqual(observer.audits[0].result, outcome.result)

    async def test_observer_failure_cannot_change_run_or_commit(self) -> None:
        observer = FailingObserver()
        store = RecordingStore()
        harness = AgentHarness(
            agent_id="failing-observer",
            model=ScriptedModel([AssistantMessage(content="done")]),
            tools=NoTools(),
            store=store,
            observers=(observer,),
        )

        outcome = await harness.run("task")
        await observer.called.wait()
        await asyncio.sleep(0)

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(harness.revision, 1)
        persisted = await store.load("failing-observer")
        assert persisted is not None
        self.assertEqual(persisted.revision, 1)
        await harness.shutdown()

    async def test_observer_runs_after_store_decision(self) -> None:
        store = RecordingStore()
        observer = StoreCheckingObserver(store)
        harness = AgentHarness(
            agent_id="observed-agent",
            model=ScriptedModel([AssistantMessage(content="done")]),
            tools=NoTools(),
            store=store,
            observers=(observer,),
        )

        await harness.run("task")
        await observer.called.wait()

        self.assertEqual(observer.persisted_revision, 1)
        await harness.shutdown()

    async def test_harness_owns_observer_lifecycle(self) -> None:
        events: list[str] = []
        observer = LifecycleObserver(events)
        harness = AgentHarness(
            agent_id="observer-lifecycle",
            model=ScriptedModel([AssistantMessage(content="done")]),
            tools=NoTools(),
            observers=(observer,),
        )

        await harness.run("task")
        await observer.called.wait()
        await harness.shutdown()

        self.assertEqual(
            events,
            ["observer:start", "observer:observe", "observer:shutdown"],
        )


if __name__ == "__main__":
    unittest.main()
