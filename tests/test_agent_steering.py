import asyncio
import tempfile
import unittest
from typing import Any

from ejagent import (
    AgentEvent,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    CompositeAgentEventSink,
    ControlStatus,
    JsonlSessionStorage,
    MethodToolHandler,
    ModelAdapter,
    ModelToolCall,
    RunStatus,
    SessionRecorder,
    SessionRecordKind,
    SteeringApplied,
    SteeringDiscarded,
    StepOutcome,
    StopReason,
    ToolCompleted,
    TurnCompleted,
)

WAIT_TOOL = {
    "type": "function",
    "function": {
        "name": "wait",
        "description": "Wait for the test to release this tool.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class BlockingFirstModel(ModelAdapter):
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = list(responses)
        self.contexts: list[tuple[dict[str, Any], ...]] = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        self.contexts.append(context.agent_messages)
        if len(self.contexts) == 1:
            self.first_started.set()
            await self.release_first.wait()
        return self.responses.pop(0)


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


class NeverCompletingModel(ModelAdapter):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        self.started.set()
        await asyncio.Future()


class BlockingToolHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((WAIT_TOOL,))
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def do_wait(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        self.started.set()
        await self.release.wait()
        return StepOutcome({"released": True})


def payloads(sink: RecordingSink) -> list[object]:
    return [event.payload for event in sink.events]


class AgentSteeringTests(unittest.IsolatedAsyncioTestCase):
    async def test_steering_during_response_forces_next_turn_in_fifo_order(
        self,
    ) -> None:
        model = BlockingFirstModel(
            [
                AssistantMessage(content="superseded answer"),
                AssistantMessage(content="corrected answer"),
            ]
        )
        sink = RecordingSink()
        agent = BaseAgent(
            model,
            agent_id="steering-fifo",
            steering_queue_capacity=2,
            event_sink=sink,
        )
        run = asyncio.create_task(agent.run(task="initial task"))
        await model.first_started.wait()

        first = await agent.steer("first correction")
        second = await agent.steer("second correction")
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(first.queue_size, 1)
        self.assertEqual(second.queue_size, 2)
        self.assertEqual(agent.pending_steering_count, 2)
        model.release_first.set()

        result = await run

        self.assertEqual(result.output, "corrected answer")
        self.assertEqual(result.turns, 2)
        self.assertEqual(agent.pending_steering_count, 0)
        self.assertEqual(
            [message.get("content") for message in model.contexts[1]],
            [
                agent.system_prompt,
                "initial task",
                "superseded answer",
                "first correction",
                "second correction",
            ],
        )
        applied = [
            payload
            for payload in payloads(sink)
            if isinstance(payload, SteeringApplied)
        ]
        self.assertEqual(
            [event.control.input_id for event in applied],
            [first.control.input_id, second.control.input_id],
        )
        self.assertEqual([event.target_turn for event in applied], [2, 2])

    async def test_queue_capacity_and_idle_submission_are_explicit(self) -> None:
        model = BlockingFirstModel(
            [AssistantMessage(content="old"), AssistantMessage(content="new")]
        )
        agent = BaseAgent(
            model,
            agent_id="steering-capacity",
            steering_queue_capacity=1,
        )

        idle = await agent.steer("too early")
        self.assertEqual(idle.status, ControlStatus.AGENT_IDLE)
        run = asyncio.create_task(agent.run(task="task"))
        await model.first_started.wait()
        accepted = await agent.steer("accepted")
        full = await agent.steer("rejected because full")
        self.assertEqual(accepted.status, ControlStatus.ACCEPTED)
        self.assertEqual(full.status, ControlStatus.QUEUE_FULL)
        self.assertEqual(full.queue_size, 1)
        model.release_first.set()
        await run

        after = await agent.steer("too late")
        self.assertEqual(after.status, ControlStatus.AGENT_IDLE)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            await agent.steer("  ")

    async def test_tool_finishes_before_steering_reaches_next_model_call(
        self,
    ) -> None:
        tool_call = ModelToolCall(id="wait-1", name="wait", arguments="{}")
        model = SequenceModel(
            [
                AssistantMessage(tool_calls=(tool_call,)),
                AssistantMessage(content="finished after steering"),
            ]
        )
        handler = BlockingToolHandler()
        observer = RecordingSink()
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            recorder = SessionRecorder(session_id="steered-tool", storage=storage)
            agent = BaseAgent(
                model,
                agent_id="steered-tool-agent",
                handlers=[handler],
                event_sink=CompositeAgentEventSink([recorder, observer]),
            )
            run = asyncio.create_task(agent.run(task="use the tool"))
            await handler.started.wait()

            receipt = await agent.steer("change direction after the tool")

            self.assertTrue(receipt.accepted)
            self.assertFalse(run.done())
            self.assertFalse(handler.release.is_set())
            handler.release.set()
            result = await run

            self.assertEqual(result.status, RunStatus.COMPLETED)
            second_context = model.contexts[1]
            roles = [message["role"] for message in second_context]
            self.assertEqual(
                roles[-4:],
                ["user", "assistant", "tool", "user"],
            )
            self.assertEqual(
                second_context[-1]["content"],
                "change direction after the tool",
            )
            observed = payloads(observer)
            tool_completed_at = next(
                index
                for index, payload in enumerate(observed)
                if isinstance(payload, ToolCompleted)
            )
            first_turn_completed_at = next(
                index
                for index, payload in enumerate(observed)
                if isinstance(payload, TurnCompleted) and payload.turn == 1
            )
            steering_at = next(
                index
                for index, payload in enumerate(observed)
                if isinstance(payload, SteeringApplied)
            )
            self.assertLess(tool_completed_at, first_turn_completed_at)
            self.assertLess(first_turn_completed_at, steering_at)

            session = await recorder.load()
            assert session is not None
            self.assertEqual(
                [message["role"] for message in session.messages],
                ["user", "assistant", "tool", "user", "assistant"],
            )
            records = await storage.records("steered-tool")
            self.assertEqual(
                [record.kind for record in records],
                [
                    SessionRecordKind.RUN_STARTED,
                    SessionRecordKind.MESSAGE_APPENDED,
                    SessionRecordKind.MESSAGES_APPENDED,
                    SessionRecordKind.STEERING_APPLIED,
                    SessionRecordKind.MESSAGE_APPENDED,
                    SessionRecordKind.RUN_FINISHED,
                ],
            )
            steering_record = records[3]
            self.assertEqual(steering_record.data["input_id"], receipt.control.input_id)
            self.assertEqual(steering_record.data["target_turn"], 2)

    async def test_cancelled_run_discards_unapplied_steering_without_persisting_it(
        self,
    ) -> None:
        model = NeverCompletingModel()
        observer = RecordingSink()
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            recorder = SessionRecorder(session_id="discarded", storage=storage)
            agent = BaseAgent(
                model,
                agent_id="discarded-agent",
                event_sink=CompositeAgentEventSink([recorder, observer]),
            )
            run = asyncio.create_task(agent.run(task="wait"))
            await model.started.wait()
            accepted = await agent.steer("never applied")
            self.assertTrue(accepted.accepted)

            self.assertTrue(agent.abort("cancel before next model call"))
            closing = await agent.steer("also too late")
            self.assertEqual(closing.status, ControlStatus.RUN_CLOSING)
            result = await run

            self.assertEqual(result.status, RunStatus.CANCELLED)
            discarded = [
                payload
                for payload in payloads(observer)
                if isinstance(payload, SteeringDiscarded)
            ]
            self.assertEqual(len(discarded), 1)
            self.assertEqual(
                discarded[0].control.input_id,
                accepted.control.input_id,
            )
            self.assertEqual(discarded[0].reason, StopReason.EXTERNAL_ABORT)
            self.assertFalse(
                any(
                    isinstance(payload, SteeringApplied)
                    for payload in payloads(observer)
                )
            )
            records = await storage.records("discarded")
            self.assertNotIn(
                SessionRecordKind.STEERING_APPLIED,
                [record.kind for record in records],
            )


if __name__ == "__main__":
    unittest.main()
