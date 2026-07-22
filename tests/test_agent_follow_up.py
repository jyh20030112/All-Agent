import asyncio
import tempfile
import unittest
from typing import Any

from ejagent import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    CompositeAgentEventSink,
    ControlStatus,
    FollowUpDiscardedError,
    FollowUpDiscardReason,
    FollowUpFailurePolicy,
    FollowUpRejectedError,
    JsonlSessionStorage,
    ModelAdapter,
    RunStatus,
    SessionRecorder,
    SessionRecordKind,
)


class BlockingSequenceModel(ModelAdapter):
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = list(responses)
        self.contexts: list[tuple[dict[str, Any], ...]] = []
        self.started = [asyncio.Event() for _ in responses]
        self.release_first = asyncio.Event()

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        index = len(self.contexts)
        self.contexts.append(context.agent_messages)
        self.started[index].set()
        if index == 0:
            await self.release_first.wait()
        return self.responses.pop(0)


class SlowFirstFinishSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.first_finish_started = asyncio.Event()
        self.release_first_finish = asyncio.Event()
        self._finish_count = 0

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)
        if isinstance(event.payload, AgentFinished):
            self._finish_count += 1
            if self._finish_count == 1:
                self.first_finish_started.set()
                await self.release_first_finish.wait()


class AgentFollowUpTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_ups_are_independent_fifo_runs_after_terminal_sink(
        self,
    ) -> None:
        model = BlockingSequenceModel(
            [
                AssistantMessage(content="initial result"),
                AssistantMessage(content="first follow-up result"),
                AssistantMessage(content="second follow-up result"),
                AssistantMessage(content="direct result"),
            ]
        )
        observer = SlowFirstFinishSink()
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            recorder = SessionRecorder(session_id="follow-up-order", storage=storage)
            agent = BaseAgent(
                model,
                agent_id="follow-up-order-agent",
                event_sink=CompositeAgentEventSink([recorder, observer]),
            )
            initial = asyncio.create_task(agent.run(task="initial"))
            await model.started[0].wait()

            first = await agent.follow_up("first follow-up")
            second = await agent.follow_up("second follow-up")
            direct = asyncio.create_task(agent.run(task="direct run"))
            self.assertTrue(first.accepted)
            self.assertTrue(second.accepted)
            self.assertEqual(first.receipt.queue_size, 1)
            self.assertEqual(second.receipt.queue_size, 2)
            self.assertEqual(agent.pending_follow_up_count, 2)

            model.release_first.set()
            await observer.first_finish_started.wait()
            idle = asyncio.create_task(agent.wait_for_idle())
            await asyncio.sleep(0)
            self.assertEqual(len(model.contexts), 1)
            self.assertFalse(initial.done())
            self.assertFalse(first.done)
            self.assertFalse(idle.done())

            observer.release_first_finish.set()
            initial_result = await initial
            first_result = await first.wait()
            second_result = await second.wait()
            direct_result = await direct
            await idle

            self.assertEqual(initial_result.output, "initial result")
            self.assertEqual(first_result.output, "first follow-up result")
            self.assertEqual(second_result.output, "second follow-up result")
            self.assertEqual(direct_result.output, "direct result")
            started_tasks = [
                event.payload.task
                for event in observer.events
                if isinstance(event.payload, AgentStarted)
            ]
            self.assertEqual(
                started_tasks,
                ["initial", "first follow-up", "second follow-up", "direct run"],
            )
            records = await storage.records("follow-up-order")
            terminal_kinds = [
                record.kind
                for record in records
                if record.kind
                in {SessionRecordKind.RUN_STARTED, SessionRecordKind.RUN_FINISHED}
            ]
            self.assertEqual(
                terminal_kinds,
                [
                    SessionRecordKind.RUN_STARTED,
                    SessionRecordKind.RUN_FINISHED,
                ]
                * 4,
            )
            run_ids = [
                record.data["run_id"]
                for record in records
                if record.kind is SessionRecordKind.RUN_STARTED
            ]
            self.assertEqual(len(set(run_ids)), 4)

    async def test_idle_queue_full_and_empty_task_are_explicit(self) -> None:
        model = BlockingSequenceModel(
            [AssistantMessage(content="initial"), AssistantMessage(content="queued")]
        )
        agent = BaseAgent(
            model,
            agent_id="follow-up-capacity",
            follow_up_queue_capacity=1,
        )

        idle = await agent.follow_up("too early")
        self.assertEqual(idle.receipt.status, ControlStatus.AGENT_IDLE)
        with self.assertRaises(FollowUpRejectedError):
            await idle.wait()

        initial = asyncio.create_task(agent.run(task="initial"))
        await model.started[0].wait()
        accepted = await agent.follow_up("accepted")
        full = await agent.follow_up("full")
        self.assertTrue(accepted.accepted)
        self.assertEqual(full.receipt.status, ControlStatus.QUEUE_FULL)
        self.assertEqual(full.receipt.queue_size, 1)
        with self.assertRaises(FollowUpRejectedError):
            await full.wait()
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            await agent.follow_up("  ")

        model.release_first.set()
        await initial
        self.assertEqual((await accepted.wait()).output, "queued")
        after = await agent.follow_up("too late")
        self.assertEqual(after.receipt.status, ControlStatus.AGENT_IDLE)

    async def test_failure_policy_discards_or_continues_pending_runs(self) -> None:
        discard_model = BlockingSequenceModel(
            [AssistantMessage(), AssistantMessage(content="must not run")]
        )
        discard_agent = BaseAgent(discard_model, agent_id="follow-up-discard")
        failed = asyncio.create_task(discard_agent.run(task="fail"))
        await discard_model.started[0].wait()
        discarded = await discard_agent.follow_up("discard me")
        discard_model.release_first.set()

        failed_result = await failed
        self.assertEqual(failed_result.status, RunStatus.FAILED)
        with self.assertRaises(FollowUpDiscardedError) as raised:
            await discarded.wait()
        self.assertEqual(
            raised.exception.reason,
            FollowUpDiscardReason.PREVIOUS_RUN_NOT_COMPLETED,
        )
        self.assertEqual(len(discard_model.contexts), 1)

        continue_model = BlockingSequenceModel(
            [AssistantMessage(), AssistantMessage(content="continued")]
        )
        continue_agent = BaseAgent(
            continue_model,
            agent_id="follow-up-continue",
            follow_up_failure_policy=FollowUpFailurePolicy.CONTINUE,
        )
        failed = asyncio.create_task(continue_agent.run(task="fail"))
        await continue_model.started[0].wait()
        continued = await continue_agent.follow_up("continue anyway")
        continue_model.release_first.set()

        self.assertEqual((await failed).status, RunStatus.FAILED)
        self.assertEqual((await continued.wait()).output, "continued")

    async def test_aborted_run_discards_follow_up_by_default(self) -> None:
        model = BlockingSequenceModel(
            [AssistantMessage(content="unused"), AssistantMessage(content="unused")]
        )
        agent = BaseAgent(model, agent_id="follow-up-abort")
        initial = asyncio.create_task(agent.run(task="initial"))
        await model.started[0].wait()
        handle = await agent.follow_up("must not run")

        self.assertTrue(agent.abort("stop the chain"))
        closing = await agent.follow_up("also must not run")
        result = await initial

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(closing.receipt.status, ControlStatus.RUN_CLOSING)
        with self.assertRaises(FollowUpDiscardedError) as raised:
            await handle.wait()
        self.assertEqual(
            raised.exception.reason,
            FollowUpDiscardReason.PREVIOUS_RUN_NOT_COMPLETED,
        )
        self.assertEqual(len(model.contexts), 1)

    async def test_shutdown_discards_queue_and_waiter_cancellation_is_local(
        self,
    ) -> None:
        model = BlockingSequenceModel(
            [AssistantMessage(content="initial"), AssistantMessage(content="unused")]
        )
        agent = BaseAgent(model, agent_id="follow-up-shutdown")
        initial = asyncio.create_task(agent.run(task="initial"))
        await model.started[0].wait()
        handle = await agent.follow_up("queued")

        waiter = asyncio.create_task(handle.wait())
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(handle.done)

        shutdown = asyncio.create_task(agent.shutdown())
        await asyncio.sleep(0)
        self.assertTrue(handle.done)
        with self.assertRaises(FollowUpDiscardedError) as raised:
            await handle.wait()
        self.assertEqual(
            raised.exception.reason,
            FollowUpDiscardReason.AGENT_SHUTDOWN,
        )
        closing = await agent.follow_up("too late")
        self.assertEqual(closing.receipt.status, ControlStatus.RUN_CLOSING)

        model.release_first.set()
        self.assertEqual((await initial).status, RunStatus.COMPLETED)
        await shutdown
        self.assertEqual(len(model.contexts), 1)
        await agent.wait_for_idle()


if __name__ == "__main__":
    unittest.main()
