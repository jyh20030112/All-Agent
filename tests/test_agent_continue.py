import asyncio
import tempfile
import unittest
from typing import Any

from simagentplg import (
    AgentContinued,
    AgentEvent,
    AgentFinished,
    AgentRunResult,
    AgentSession,
    AgentStarted,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    CompositeAgentEventSink,
    ContinueRejectedError,
    ContinueRejectedReason,
    FollowUpDiscardedError,
    FollowUpDiscardReason,
    JsonlSessionStorage,
    ModelAdapter,
    RunStatus,
    SessionRecorder,
    SessionRecordKind,
    SessionRunIntent,
    StopReason,
)


class SequenceModel(ModelAdapter):
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = list(responses)
        self.contexts: list[tuple[dict[str, Any], ...]] = []

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        self.contexts.append(context.agent_messages)
        return self.responses.pop(0)


class BlockingModel(SequenceModel):
    def __init__(
        self,
        responses: list[AssistantMessage],
        *,
        block_call: int,
    ) -> None:
        super().__init__(responses)
        self.block_call = block_call
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        index = len(self.contexts)
        self.contexts.append(context.agent_messages)
        if index == self.block_call:
            self.started.set()
            await self.release.wait()
        return self.responses.pop(0)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class AgentContinueTests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_is_a_distinct_run_without_new_user_message(self) -> None:
        model = SequenceModel(
            [
                AssistantMessage(content="initial answer"),
                AssistantMessage(content="continued answer"),
            ]
        )
        observer = RecordingSink()
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            recorder = SessionRecorder(session_id="continue-basic", storage=storage)
            agent = BaseAgent(
                model,
                agent_id="continue-basic-agent",
                event_sink=CompositeAgentEventSink([recorder, observer]),
            )

            initial = await agent.run(task="investigate")
            history_before_continue = tuple(dict(message) for message in agent.messages)
            self.assertTrue(agent.can_continue)
            continued = await agent.continue_run()

            self.assertEqual(initial.output, "initial answer")
            self.assertEqual(continued.output, "continued answer")
            self.assertIsNone(agent.state.task)
            self.assertEqual(model.contexts[1], history_before_continue)
            self.assertEqual(
                [message["role"] for message in agent.messages],
                ["system", "user", "assistant", "assistant"],
            )

            starts = [
                event
                for event in observer.events
                if isinstance(event.payload, (AgentStarted, AgentContinued))
            ]
            self.assertEqual(len(starts), 2)
            self.assertIsInstance(starts[0].payload, AgentStarted)
            self.assertIsInstance(starts[1].payload, AgentContinued)
            self.assertNotEqual(starts[0].run_id, starts[1].run_id)

            session = await recorder.load()
            assert session is not None
            self.assertEqual(
                [run.intent for run in session.runs],
                [SessionRunIntent.TASK, SessionRunIntent.CONTINUE],
            )
            self.assertEqual([run.task for run in session.runs], ["investigate", None])
            self.assertEqual(
                [message["role"] for message in session.messages],
                ["user", "assistant", "assistant"],
            )
            records = await storage.records("continue-basic")
            self.assertEqual(
                [record.kind for record in records],
                [
                    SessionRecordKind.RUN_STARTED,
                    SessionRecordKind.MESSAGE_APPENDED,
                    SessionRecordKind.RUN_FINISHED,
                    SessionRecordKind.RUN_CONTINUED,
                    SessionRecordKind.MESSAGE_APPENDED,
                    SessionRecordKind.RUN_FINISHED,
                ],
            )

    async def test_continue_rejection_reasons_are_explicit(self) -> None:
        empty = BaseAgent(SequenceModel([]), agent_id="continue-empty")
        self.assertEqual(
            empty.continue_rejection_reason,
            ContinueRejectedReason.NO_PREVIOUS_RUN,
        )
        with self.assertRaises(ContinueRejectedError) as raised:
            await empty.continue_run()
        self.assertEqual(
            raised.exception.reason, ContinueRejectedReason.NO_PREVIOUS_RUN
        )
        with self.assertRaises(ContinueRejectedError) as raised:
            await empty.orchestrator.continue_run()
        self.assertEqual(
            raised.exception.reason, ContinueRejectedReason.NO_PREVIOUS_RUN
        )

        active_model = BlockingModel(
            [AssistantMessage(content="cancelled")],
            block_call=0,
        )
        active = BaseAgent(active_model, agent_id="continue-active")
        run = asyncio.create_task(active.run(task="block"))
        await active_model.started.wait()
        with self.assertRaises(ContinueRejectedError) as raised:
            await active.continue_run()
        self.assertEqual(raised.exception.reason, ContinueRejectedReason.AGENT_ACTIVE)

        self.assertTrue(active.abort())
        cancelled = await run
        self.assertEqual(cancelled.status, RunStatus.CANCELLED)
        with self.assertRaises(ContinueRejectedError) as raised:
            await active.continue_run()
        self.assertEqual(
            raised.exception.reason,
            ContinueRejectedReason.UNSUPPORTED_STOP_REASON,
        )

        incomplete = AgentSession(session_id="incomplete")
        incomplete.bind_agent("continue-incomplete")
        incomplete.begin_run("run-1", "task", 1)
        incomplete.append_message(
            "run-1",
            2,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "missing-result",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
        )
        incomplete.finish_run(
            "run-1",
            3,
            AgentRunResult(
                status=RunStatus.COMPLETED,
                stop_reason=StopReason.TEXT_RESPONSE,
                turns=1,
                output="incomplete",
            ),
        )
        restored = BaseAgent(SequenceModel([]), agent_id="continue-incomplete")
        restored.restore_session(incomplete)
        self.assertEqual(
            restored.continue_rejection_reason,
            ContinueRejectedReason.INCOMPLETE_TOOL_STATE,
        )

    async def test_steering_reaches_continue_safe_point(self) -> None:
        model = BlockingModel(
            [
                AssistantMessage(content="initial"),
                AssistantMessage(content="superseded continuation"),
                AssistantMessage(content="steered continuation"),
            ],
            block_call=1,
        )
        agent = BaseAgent(model, agent_id="continue-steering")
        await agent.run(task="initial task")

        continuing = asyncio.create_task(agent.continue_run())
        await model.started.wait()
        receipt = await agent.steer("continue in a safer direction")
        self.assertTrue(receipt.accepted)
        model.release.set()
        result = await continuing

        self.assertEqual(result.output, "steered continuation")
        self.assertEqual(result.turns, 2)
        self.assertEqual(model.contexts[2][-1]["role"], "user")
        self.assertEqual(
            model.contexts[2][-1]["content"],
            "continue in a safer direction",
        )

    async def test_restored_session_can_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            first_recorder = SessionRecorder(
                session_id="continue-restore", storage=storage
            )
            first = BaseAgent(
                SequenceModel([AssistantMessage(content="saved")]),
                agent_id="continue-restore-agent",
                event_sink=first_recorder,
            )
            await first.run(task="persist this")
            saved = await first_recorder.load()
            assert saved is not None

            second_recorder = SessionRecorder(
                session_id="continue-restore",
                storage=storage,
            )
            model = SequenceModel([AssistantMessage(content="resumed")])
            restored = BaseAgent(
                model,
                agent_id="continue-restore-agent",
                event_sink=second_recorder,
            )
            restored.restore_session(saved)
            self.assertTrue(restored.can_continue)

            result = await restored.continue_run()

            self.assertEqual(result.output, "resumed")
            self.assertEqual(
                [message["role"] for message in model.contexts[0]],
                ["system", "user", "assistant"],
            )
            reloaded = await second_recorder.load()
            assert reloaded is not None
            self.assertEqual(len(reloaded.runs), 2)
            self.assertEqual(reloaded.runs[-1].intent, SessionRunIntent.CONTINUE)

    async def test_safe_limit_failure_can_continue(self) -> None:
        session = AgentSession(session_id="continue-limit")
        session.bind_agent("continue-limit-agent")
        session.begin_run("limited-run", "long task", 1)
        session.append_message(
            "limited-run",
            2,
            {"role": "assistant", "content": "partial progress"},
        )
        session.finish_run(
            "limited-run",
            3,
            AgentRunResult(
                status=RunStatus.FAILED,
                stop_reason=StopReason.MAX_STEPS,
                turns=1,
                error="step limit reached",
            ),
        )
        model = SequenceModel([AssistantMessage(content="finished")])
        agent = BaseAgent(model, agent_id="continue-limit-agent")
        agent.restore_session(session)

        self.assertTrue(agent.can_continue)
        self.assertEqual((await agent.continue_run()).output, "finished")

    async def test_continue_wait_for_idle_includes_terminal_sink(self) -> None:
        model = BlockingModel(
            [AssistantMessage(content="first"), AssistantMessage(content="second")],
            block_call=1,
        )
        observer = RecordingSink()
        agent = BaseAgent(model, agent_id="continue-idle", event_sink=observer)
        await agent.run(task="first")
        continuing = asyncio.create_task(agent.continue_run())
        await model.started.wait()

        idle = asyncio.create_task(agent.wait_for_idle())
        await asyncio.sleep(0)
        self.assertFalse(idle.done())
        model.release.set()
        await continuing
        await idle
        self.assertIsInstance(observer.events[-1].payload, AgentFinished)

    async def test_continue_preserves_follow_up_and_shutdown_semantics(self) -> None:
        model = BlockingModel(
            [
                AssistantMessage(content="first"),
                AssistantMessage(content="continued"),
                AssistantMessage(content="follow-up"),
            ],
            block_call=1,
        )
        agent = BaseAgent(model, agent_id="continue-chain")
        await agent.run(task="first")
        continuing = asyncio.create_task(agent.continue_run())
        await model.started.wait()
        follow_up = await agent.follow_up("after continue")
        self.assertTrue(follow_up.accepted)

        shutdown = asyncio.create_task(agent.shutdown())
        await asyncio.sleep(0)
        with self.assertRaises(ContinueRejectedError) as raised:
            await agent.continue_run()
        self.assertEqual(
            raised.exception.reason,
            ContinueRejectedReason.AGENT_SHUTTING_DOWN,
        )
        model.release.set()

        self.assertEqual((await continuing).output, "continued")
        with self.assertRaises(FollowUpDiscardedError) as discarded:
            await follow_up.wait()
        self.assertEqual(
            discarded.exception.reason,
            FollowUpDiscardReason.AGENT_SHUTDOWN,
        )
        await shutdown
        self.assertEqual(len(model.contexts), 2)

    async def test_follow_up_runs_after_successful_continue(self) -> None:
        model = BlockingModel(
            [
                AssistantMessage(content="first"),
                AssistantMessage(content="continued"),
                AssistantMessage(content="followed up"),
            ],
            block_call=1,
        )
        agent = BaseAgent(model, agent_id="continue-follow-up")
        await agent.run(task="first")
        continuing = asyncio.create_task(agent.continue_run())
        await model.started.wait()
        follow_up = await agent.follow_up("next task")
        model.release.set()

        self.assertEqual((await continuing).output, "continued")
        self.assertEqual((await follow_up.wait()).output, "followed up")
        self.assertEqual(model.contexts[2][-1]["role"], "user")
        self.assertEqual(model.contexts[2][-1]["content"], "next task")


if __name__ == "__main__":
    unittest.main()
