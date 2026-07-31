from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from ejagent.contracts import (
    AssistantMessage,
    CancellationToken,
    FailureCode,
    ModelCallError,
    ModelPort,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    RunStatus,
    SessionCommit,
    SessionConflictError,
    SessionSnapshot,
    SessionStore,
    SessionStoreError,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutor,
    UserMessage,
)
from ejagent.harness import (
    AgentHarness,
    HarnessClosedError,
    HarnessStatus,
    MemorySessionStore,
)


def fixed_clock() -> datetime:
    return datetime(2026, 7, 31, 14, 0, tzinfo=UTC)


def ids(*values: str) -> Iterable[str]:
    return iter(values)


class ScriptedModel(ModelPort):
    def __init__(
        self,
        responses: Sequence[AssistantMessage | ModelCallError],
    ) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, ModelCallError):
            raise response
        yield ModelResponseCompleted(response)


class BlockingModel(ModelPort):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.started.set()
        await asyncio.Event().wait()
        yield ModelResponseCompleted(AssistantMessage(content="unreachable"))


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


class FailingStore(SessionStore):
    async def load(self, agent_id: str) -> SessionSnapshot | None:
        return None

    async def commit(self, commit: SessionCommit) -> SessionSnapshot:
        raise SessionStoreError("durable backend unavailable")


class RecordingResource:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_start: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_start = fail_start

    async def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(f"cannot start {self.name}")

    async def shutdown(self) -> None:
        self.events.append(f"shutdown:{self.name}")


class ManagedScriptedModel(ScriptedModel):
    def __init__(self, events: list[str]) -> None:
        super().__init__([AssistantMessage(content="managed")])
        self.events = events

    async def start(self) -> None:
        self.events.append("start:model")

    async def shutdown(self) -> None:
        self.events.append("shutdown:model")


class ManagedNoTools(NoTools):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start(self) -> None:
        self.events.append("start:tools")

    async def shutdown(self) -> None:
        self.events.append("shutdown:tools")


class AgentHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_run_commits_only_after_store_accepts(self) -> None:
        store = MemorySessionStore()
        model = ScriptedModel([AssistantMessage(content="done")])
        run_ids = ids("run-1")
        harness = AgentHarness(
            agent_id="agent-1",
            model=model,
            tools=NoTools(),
            initial_messages=(SystemMessage("be precise"),),
            store=store,
            run_id_factory=lambda: next(run_ids),
            clock=fixed_clock,
        )

        outcome = await harness.run("solve it")

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(harness.revision, 1)
        self.assertEqual(
            harness.messages,
            (
                SystemMessage("be precise"),
                UserMessage("solve it"),
                AssistantMessage(content="done"),
            ),
        )
        self.assertEqual(harness.last_result, outcome.result)
        self.assertEqual((await store.load("agent-1")), harness.snapshot)
        self.assertEqual(len(await store.commits("agent-1")), 1)
        await harness.shutdown()

    async def test_failed_run_is_audited_without_advancing_conversation(self) -> None:
        store = MemorySessionStore()
        model = ScriptedModel(
            [
                ModelCallError(
                    FailureCode.RATE_LIMIT,
                    "slow down",
                    retryable=True,
                )
            ]
        )
        harness = AgentHarness(
            agent_id="agent-failed",
            model=model,
            tools=NoTools(),
            initial_messages=(SystemMessage("stable"),),
            store=store,
            run_id_factory=lambda: "failed-run",
            clock=fixed_clock,
        )

        outcome = await harness.run("transient task")

        self.assertEqual(outcome.result.status, RunStatus.FAILED)
        self.assertEqual(harness.revision, 0)
        self.assertEqual(harness.messages, (SystemMessage("stable"),))
        commits = await store.commits("agent-failed")
        self.assertEqual(len(commits), 1)
        self.assertFalse(commits[0].advances_revision)
        self.assertEqual(commits[0].outcome.failure, outcome.failure)

    async def test_store_failure_cannot_report_success_or_advance_state(self) -> None:
        harness = AgentHarness(
            agent_id="agent-store-failure",
            model=ScriptedModel([AssistantMessage(content="model succeeded")]),
            tools=NoTools(),
            initial_messages=(SystemMessage("unchanged"),),
            store=FailingStore(),
            run_id_factory=lambda: "uncommitted-run",
            clock=fixed_clock,
        )

        outcome = await harness.run("do work")

        self.assertEqual(outcome.result.status, RunStatus.FAILED)
        self.assertEqual(outcome.result.stop_reason, StopReason.PERSISTENCE_FAILED)
        assert outcome.failure is not None
        self.assertEqual(outcome.failure.phase.value, "commit")
        self.assertEqual(outcome.failure.code, FailureCode.PERSISTENCE_FAILED)
        self.assertEqual(harness.revision, 0)
        self.assertEqual(harness.messages, (SystemMessage("unchanged"),))
        self.assertEqual(outcome.audit_records[-1].kind, "commit_failed")

    async def test_concurrent_calls_are_fifo_and_use_fresh_revisions(self) -> None:
        model = ScriptedModel(
            [
                AssistantMessage(content="first result"),
                AssistantMessage(content="second result"),
            ]
        )
        run_ids = ids("run-1", "run-2")
        harness = AgentHarness(
            agent_id="serial-agent",
            model=model,
            tools=NoTools(),
            run_id_factory=lambda: next(run_ids),
            clock=fixed_clock,
        )

        first = asyncio.create_task(harness.run("first"))
        await asyncio.sleep(0)
        second = asyncio.create_task(harness.run("second"))
        first_outcome, second_outcome = await asyncio.gather(first, second)

        self.assertEqual(first_outcome.delta.base_revision, 0)
        self.assertEqual(second_outcome.delta.base_revision, 1)
        self.assertEqual(harness.revision, 2)
        self.assertEqual(
            model.requests[1].messages,
            (
                UserMessage("first"),
                AssistantMessage(content="first result"),
                UserMessage("second"),
            ),
        )

    async def test_continue_run_uses_history_without_new_user_message(self) -> None:
        store = MemorySessionStore()
        model = ScriptedModel(
            [
                AssistantMessage(content="initial"),
                AssistantMessage(content="continued"),
            ]
        )
        run_ids = ids("task-run", "continue-run")
        harness = AgentHarness(
            agent_id="continue-agent",
            model=model,
            tools=NoTools(),
            store=store,
            run_id_factory=lambda: next(run_ids),
            clock=fixed_clock,
        )

        await harness.run("begin")
        outcome = await harness.continue_run()

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(
            outcome.delta.messages,
            (AssistantMessage(content="continued"),),
        )
        commits = await store.commits("continue-agent")
        self.assertEqual(commits[-1].outcome.delta.base_revision, 1)
        self.assertEqual(commits[-1].outcome.result.run_id, "continue-run")

    async def test_cancellation_is_structured_and_does_not_commit_delta(self) -> None:
        model = BlockingModel()
        harness = AgentHarness(
            agent_id="cancel-agent",
            model=model,
            tools=NoTools(),
            initial_messages=(SystemMessage("retain me"),),
            run_id_factory=lambda: "cancelled-run",
            clock=fixed_clock,
        )
        running = asyncio.create_task(harness.run("never finish"))
        await model.started.wait()

        self.assertTrue(harness.cancel("user stopped it"))
        self.assertFalse(harness.cancel("second reason"))
        outcome = await running

        self.assertEqual(outcome.result.status, RunStatus.CANCELLED)
        self.assertEqual(harness.revision, 0)
        self.assertEqual(harness.messages, (SystemMessage("retain me"),))
        self.assertEqual(harness.status, HarnessStatus.READY)

    async def test_startup_rolls_back_resources_in_reverse_order(self) -> None:
        events: list[str] = []
        resources = (
            RecordingResource("one", events),
            RecordingResource("two", events),
            RecordingResource("broken", events, fail_start=True),
        )
        harness = AgentHarness(
            agent_id="resource-agent",
            model=ScriptedModel([AssistantMessage(content="unused")]),
            tools=NoTools(),
            resources=resources,
        )

        with self.assertRaisesRegex(RuntimeError, "cannot start broken"):
            await harness.start()

        self.assertEqual(
            events,
            [
                "start:one",
                "start:two",
                "start:broken",
                "shutdown:two",
                "shutdown:one",
            ],
        )
        self.assertEqual(harness.status, HarnessStatus.NEW)

    async def test_harness_owns_model_and_tool_lifecycle_without_duplicates(
        self,
    ) -> None:
        events: list[str] = []
        model = ManagedScriptedModel(events)
        tools = ManagedNoTools(events)
        harness = AgentHarness(
            agent_id="managed-dependencies",
            model=model,
            tools=tools,
            resources=(model, tools),
            run_id_factory=lambda: "managed-run",
            clock=fixed_clock,
        )

        await harness.run("execute")
        await harness.shutdown()

        self.assertEqual(
            events,
            [
                "start:model",
                "start:tools",
                "shutdown:tools",
                "shutdown:model",
            ],
        )

    async def test_shutdown_cancels_run_and_closes_resources_in_reverse(self) -> None:
        events: list[str] = []
        model = BlockingModel()
        harness = AgentHarness(
            agent_id="shutdown-agent",
            model=model,
            tools=NoTools(),
            resources=(
                RecordingResource("one", events),
                RecordingResource("two", events),
            ),
            run_id_factory=lambda: "shutdown-run",
            clock=fixed_clock,
        )
        running = asyncio.create_task(harness.run("wait"))
        await model.started.wait()

        await harness.shutdown()
        outcome = await running

        self.assertEqual(outcome.result.status, RunStatus.CANCELLED)
        self.assertEqual(harness.status, HarnessStatus.CLOSED)
        self.assertEqual(
            events,
            ["start:one", "start:two", "shutdown:two", "shutdown:one"],
        )
        with self.assertRaises(HarnessClosedError):
            await harness.run("too late")

    async def test_new_harness_restores_the_committed_snapshot(self) -> None:
        store = MemorySessionStore()
        first = AgentHarness(
            agent_id="durable-agent",
            model=ScriptedModel([AssistantMessage(content="saved")]),
            tools=NoTools(),
            store=store,
            run_id_factory=lambda: "saved-run",
            clock=fixed_clock,
        )
        await first.run("persist")
        await first.shutdown()

        second_model = ScriptedModel([AssistantMessage(content="restored")])
        second = AgentHarness(
            agent_id="durable-agent",
            model=second_model,
            tools=NoTools(),
            store=store,
            run_id_factory=lambda: "restored-run",
            clock=fixed_clock,
        )

        outcome = await second.continue_run()

        self.assertEqual(outcome.delta.base_revision, 1)
        self.assertEqual(second.revision, 2)
        self.assertEqual(
            second_model.requests[0].messages,
            (UserMessage("persist"), AssistantMessage(content="saved")),
        )

    async def test_memory_store_repeats_identical_commit_idempotently(self) -> None:
        store = MemorySessionStore()
        harness = AgentHarness(
            agent_id="idempotent-agent",
            model=ScriptedModel([AssistantMessage(content="once")]),
            tools=NoTools(),
            store=store,
            run_id_factory=lambda: "stable-run-id",
            clock=fixed_clock,
        )
        await harness.run("commit")
        commit = (await store.commits("idempotent-agent"))[0]

        repeated = await store.commit(commit)

        self.assertEqual(repeated, harness.snapshot)
        self.assertEqual(len(await store.commits("idempotent-agent")), 1)

    async def test_memory_store_rejects_reused_run_id_with_new_content(self) -> None:
        store = MemorySessionStore()
        harness = AgentHarness(
            agent_id="reuse-agent",
            model=ScriptedModel([AssistantMessage(content="original")]),
            tools=NoTools(),
            store=store,
            run_id_factory=lambda: "reused-id",
            clock=fixed_clock,
        )
        await harness.run("commit")
        commit = (await store.commits("reuse-agent"))[0]
        changed_result = replace(commit.outcome.result, output="changed")
        changed_outcome = replace(commit.outcome, result=changed_result)
        changed_commit = replace(commit, outcome=changed_outcome)

        with self.assertRaises(SessionConflictError):
            await store.commit(changed_commit)

    async def test_stale_harness_commit_becomes_persistence_failure(self) -> None:
        store = MemorySessionStore()
        first = AgentHarness(
            agent_id="shared-agent",
            model=ScriptedModel([AssistantMessage(content="winner")]),
            tools=NoTools(),
            store=store,
            run_id_factory=lambda: "winner-run",
            clock=fixed_clock,
        )
        stale = AgentHarness(
            agent_id="shared-agent",
            model=ScriptedModel([AssistantMessage(content="stale")]),
            tools=NoTools(),
            store=store,
            run_id_factory=lambda: "stale-run",
            clock=fixed_clock,
        )
        await first.start()
        await stale.start()

        await first.run("first writer")
        outcome = await stale.run("stale writer")

        self.assertEqual(outcome.result.status, RunStatus.FAILED)
        self.assertEqual(outcome.result.stop_reason, StopReason.PERSISTENCE_FAILED)
        self.assertEqual(stale.revision, 0)
        persisted = await store.load("shared-agent")
        assert persisted is not None
        self.assertEqual(persisted.revision, 1)
        self.assertIn(AssistantMessage(content="winner"), persisted.messages)


if __name__ == "__main__":
    unittest.main()
