import unittest
from dataclasses import FrozenInstanceError
from typing import Any, get_type_hints

from ejagent import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    MessageCompleted,
    MethodToolHandler,
    ModelAdapter,
    ModelToolCall,
    RunStatus,
    RuntimePolicy,
    StepOutcome,
    StopReason,
    ToolCallResult,
    ToolCompleted,
    ToolControl,
    ToolStarted,
    TurnCompleted,
    TurnStarted,
)

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Return the provided text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Settle the current task.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
}


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class FailingEventSink:
    def __init__(self) -> None:
        self.calls = 0

    async def emit(self, event: AgentEvent) -> None:
        self.calls += 1
        raise RuntimeError(f"sink failed for {event.kind}")


class SequenceModel(ModelAdapter):
    def __init__(
        self,
        responses: list[AssistantMessage | Exception],
    ) -> None:
        self.responses = list(responses)

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class EchoHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((ECHO_TOOL,))
        self.calls = 0

    async def do_echo(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        self.calls += 1
        return StepOutcome({"text": arguments.get("text")})


class FinishHandler(MethodToolHandler):
    def __init__(self, control: ToolControl) -> None:
        super().__init__((FINISH_TOOL,))
        self.control = control

    async def do_finish(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome(
            {"summary": arguments.get("summary")},
            control=self.control,
        )


def event_payloads(sink: RecordingEventSink) -> list[object]:
    return [event.payload for event in sink.events]


class AgentEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_run_emits_ordered_correlated_events(self) -> None:
        response = AssistantMessage(content="done")
        sink = RecordingEventSink()
        agent = BaseAgent(
            SequenceModel([response]),
            agent_id="text-events",
            event_sink=sink,
        )

        result = await agent.run(task="complete")

        payloads = event_payloads(sink)
        self.assertEqual(
            [type(payload) for payload in payloads],
            [
                AgentStarted,
                TurnStarted,
                MessageCompleted,
                TurnCompleted,
                AgentFinished,
            ],
        )
        self.assertIs(payloads[2].message, response)
        self.assertIs(payloads[-1].result, result)
        self.assertEqual(
            [event.sequence for event in sink.events],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            {event.agent_id for event in sink.events},
            {"text-events"},
        )
        self.assertEqual(len({event.run_id for event in sink.events}), 1)

        with self.assertRaises(FrozenInstanceError):
            sink.events[0].sequence = 99  # type: ignore[misc]
        self.assertIs(
            get_type_hints(ToolCompleted)["result"],
            ToolCallResult,
        )

    async def test_tool_run_emits_tool_events_between_message_and_turn(self) -> None:
        tool_call = ModelToolCall(
            id="call-1",
            name="echo",
            arguments='{"text": "hello"}',
        )
        sink = RecordingEventSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(tool_calls=(tool_call,)),
                    AssistantMessage(content="finished"),
                ]
            ),
            agent_id="tool-events",
            handlers=[EchoHandler()],
            event_sink=sink,
        )

        result = await agent.run(task="use echo")

        payloads = event_payloads(sink)
        self.assertEqual(
            [type(payload) for payload in payloads],
            [
                AgentStarted,
                TurnStarted,
                MessageCompleted,
                ToolStarted,
                ToolCompleted,
                TurnCompleted,
                TurnStarted,
                MessageCompleted,
                TurnCompleted,
                AgentFinished,
            ],
        )
        self.assertIs(payloads[3].tool_call, tool_call)
        self.assertIs(payloads[4].tool_call, tool_call)
        self.assertIsNone(payloads[4].result.error)
        self.assertEqual(result.status, RunStatus.COMPLETED)

    async def test_terminal_tool_controls_reuse_agent_run_result(self) -> None:
        statuses = {
            ToolControl.COMPLETE: RunStatus.COMPLETED,
            ToolControl.REJECT: RunStatus.REJECTED,
            ToolControl.CANCEL: RunStatus.CANCELLED,
        }

        for control, expected_status in statuses.items():
            with self.subTest(control=control):
                sink = RecordingEventSink()
                tool_call = ModelToolCall(
                    id=f"call-{control}",
                    name="finish",
                    arguments='{"summary": "settled"}',
                )
                agent = BaseAgent(
                    SequenceModel([AssistantMessage(tool_calls=(tool_call,))]),
                    agent_id=f"terminal-{control}",
                    handlers=[FinishHandler(control)],
                    event_sink=sink,
                )

                result = await agent.run(task="settle")

                tool_event = next(
                    payload
                    for payload in event_payloads(sink)
                    if isinstance(payload, ToolCompleted)
                )
                finish_event = event_payloads(sink)[-1]
                self.assertEqual(tool_event.result.control, control)
                self.assertIsInstance(finish_event, AgentFinished)
                self.assertIs(finish_event.result, result)
                self.assertEqual(result.status, expected_status)

    async def test_provider_failure_still_completes_turn_and_run_events(self) -> None:
        sink = RecordingEventSink()
        agent = BaseAgent(
            SequenceModel([RuntimeError("provider unavailable")]),
            agent_id="provider-failure",
            event_sink=sink,
        )

        result = await agent.run(task="fail")

        self.assertEqual(
            [type(payload) for payload in event_payloads(sink)],
            [AgentStarted, TurnStarted, TurnCompleted, AgentFinished],
        )
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.stop_reason, StopReason.RUNTIME_ERROR)

    async def test_tool_argument_error_is_visible_in_tool_completed(self) -> None:
        sink = RecordingEventSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=(
                            ModelToolCall(
                                id="invalid-json",
                                name="echo",
                                arguments="[]",
                            ),
                        )
                    ),
                    AssistantMessage(content="recovered"),
                ]
            ),
            agent_id="tool-error",
            handlers=[EchoHandler()],
            event_sink=sink,
        )

        result = await agent.run(task="recover")

        completed = next(
            payload
            for payload in event_payloads(sink)
            if isinstance(payload, ToolCompleted)
        )
        self.assertIn("JSON object", completed.result.error or "")
        self.assertEqual(result.status, RunStatus.COMPLETED)

    async def test_repeated_tool_failure_keeps_tool_and_turn_events_paired(
        self,
    ) -> None:
        call = ModelToolCall(
            id="repeated",
            name="echo",
            arguments='{"text": "same"}',
        )
        sink = RecordingEventSink()
        handler = EchoHandler()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(tool_calls=(call,)),
                    AssistantMessage(tool_calls=(call,)),
                ]
            ),
            agent_id="repeated-events",
            handlers=[handler],
            runtime_policy=RuntimePolicy(
                max_steps=2,
                max_repeated_tool_calls=2,
            ),
            event_sink=sink,
        )

        result = await agent.run(task="repeat")

        payloads = event_payloads(sink)
        started = [p for p in payloads if isinstance(p, ToolStarted)]
        completed = [p for p in payloads if isinstance(p, ToolCompleted)]
        turns_started = [p for p in payloads if isinstance(p, TurnStarted)]
        turns_completed = [p for p in payloads if isinstance(p, TurnCompleted)]
        self.assertEqual(len(started), len(completed))
        self.assertEqual(len(turns_started), len(turns_completed))
        self.assertIn("consecutive times", completed[-1].result.error or "")
        self.assertEqual(handler.calls, 1)
        self.assertEqual(result.stop_reason, StopReason.REPEATED_TOOL_CALL)

    async def test_sink_failure_does_not_change_agent_result(self) -> None:
        sink = FailingEventSink()
        agent = BaseAgent(
            SequenceModel([AssistantMessage(content="done")]),
            agent_id="sink-failure",
            event_sink=sink,
        )

        with self.assertLogs("sink-failure", level="WARNING"):
            result = await agent.run(task="complete")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(sink.calls, 5)

    async def test_each_run_has_a_new_id_and_sequence(self) -> None:
        sink = RecordingEventSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(content="first"),
                    AssistantMessage(content="second"),
                ]
            ),
            agent_id="multiple-runs",
            event_sink=sink,
        )

        await agent.run(task="first")
        await agent.run(task="second")

        first_run_id = sink.events[0].run_id
        second_run_id = sink.events[5].run_id
        self.assertNotEqual(first_run_id, second_run_id)
        self.assertEqual(
            [event.sequence for event in sink.events[:5]],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [event.sequence for event in sink.events[5:]],
            [1, 2, 3, 4, 5],
        )


if __name__ == "__main__":
    unittest.main()
