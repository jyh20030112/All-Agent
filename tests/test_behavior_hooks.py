import asyncio
import tempfile
import unittest
from typing import Any

from ejagent import (
    AgentEvent,
    AgentFinished,
    AssistantMessage,
    BaseAgent,
    BehaviorDecision,
    CancellationToken,
    CompositeAgentEventSink,
    JsonlSessionStorage,
    MessageCompleted,
    MethodToolHandler,
    ModelAdapter,
    ModelToolCall,
    RunStatus,
    SessionRecorder,
    StepOutcome,
    StopReason,
    ToolCompleted,
    TurnCompleted,
    TurnSnapshot,
)

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Return a test value.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}


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


class EchoHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((ECHO_TOOL,))

    async def do_echo(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome({"value": arguments["value"]})


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class StopHook:
    name = "stop-guard"

    def __init__(self, sink: RecordingSink | None = None) -> None:
        self.sink = sink
        self.snapshots: list[TurnSnapshot] = []

    async def after_turn(
        self,
        snapshot: TurnSnapshot,
        *,
        cancellation: CancellationToken,
    ) -> BehaviorDecision:
        if self.sink is not None:
            assert isinstance(self.sink.events[-1].payload, TurnCompleted)
        self.snapshots.append(snapshot)
        detached = snapshot.messages
        detached[-1]["content"] = "mutated outside AgentState"
        return BehaviorDecision.stop("stopped by behavior policy")


class OrderedHook:
    def __init__(
        self,
        name: str,
        calls: list[str],
        decision: BehaviorDecision | None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.decision = decision
        self.last_content: Any = None

    async def after_turn(
        self,
        snapshot: TurnSnapshot,
        *,
        cancellation: CancellationToken,
    ) -> BehaviorDecision | None:
        self.calls.append(self.name)
        self.last_content = snapshot.messages[-1]["content"]
        return self.decision


class RaisingHook:
    name = "broken-hook"

    async def after_turn(
        self,
        snapshot: TurnSnapshot,
        *,
        cancellation: CancellationToken,
    ) -> BehaviorDecision:
        raise ValueError("hook exploded")


class InvalidHook:
    async def after_turn(
        self,
        snapshot: TurnSnapshot,
        *,
        cancellation: CancellationToken,
    ) -> Any:
        return "invalid"


class BlockingHook:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.interrupted = asyncio.Event()

    async def after_turn(
        self,
        snapshot: TurnSnapshot,
        *,
        cancellation: CancellationToken,
    ) -> BehaviorDecision:
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.interrupted.set()


def tool_turn(call_id: str = "echo-1") -> AssistantMessage:
    return AssistantMessage(
        tool_calls=(
            ModelToolCall(
                id=call_id,
                name="echo",
                arguments='{"value": "ready"}',
            ),
        )
    )


class BehaviorHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_occurs_after_tool_result_and_persists_terminal_result(
        self,
    ) -> None:
        model = SequenceModel(
            [tool_turn(), AssistantMessage(content="must not be requested")]
        )
        observer = RecordingSink()
        hook = StopHook(observer)
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            recorder = SessionRecorder(session_id="behavior-stop", storage=storage)
            agent = BaseAgent(
                model,
                agent_id="behavior-stop-agent",
                handlers=[EchoHandler()],
                behavior_hooks=[hook],
                event_sink=CompositeAgentEventSink([recorder, observer]),
            )

            result = await agent.run(task="use the tool")

            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual(result.stop_reason, StopReason.BEHAVIOR_STOP)
            self.assertEqual(result.output, "stopped by behavior policy")
            self.assertEqual(result.turns, 1)
            self.assertTrue(agent.can_continue)
            self.assertEqual(len(model.contexts), 1)
            self.assertEqual(len(hook.snapshots), 1)
            snapshot = hook.snapshots[0]
            self.assertEqual(snapshot.agent_id, agent.agent_id)
            self.assertEqual(snapshot.turn, 1)
            self.assertEqual(snapshot.task, "use the tool")
            self.assertEqual(
                [message["role"] for message in snapshot.messages[-3:]],
                ["user", "assistant", "tool"],
            )
            self.assertNotEqual(
                agent.messages[-1]["content"],
                "mutated outside AgentState",
            )

            payloads = [event.payload for event in observer.events]
            tool_completed = next(
                index
                for index, payload in enumerate(payloads)
                if isinstance(payload, ToolCompleted)
            )
            turn_completed = next(
                index
                for index, payload in enumerate(payloads)
                if isinstance(payload, TurnCompleted)
            )
            self.assertLess(tool_completed, turn_completed)
            self.assertIsInstance(payloads[-1], AgentFinished)
            session = await recorder.load()
            assert session is not None
            assert session.runs[-1].result is not None
            self.assertEqual(
                session.runs[-1].result.stop_reason,
                StopReason.BEHAVIOR_STOP,
            )

    async def test_hooks_are_ordered_and_stop_short_circuits_remaining_hooks(
        self,
    ) -> None:
        calls: list[str] = []
        first = OrderedHook("first", calls, None)
        second = OrderedHook("second", calls, BehaviorDecision.stop())
        third = OrderedHook("third", calls, BehaviorDecision.continue_run())
        agent = BaseAgent(
            SequenceModel([tool_turn(), AssistantMessage(content="unused")]),
            agent_id="behavior-order",
            handlers=[EchoHandler()],
            behavior_hooks=[first, second, third],
        )

        result = await agent.run(task="ordered")

        self.assertEqual(result.stop_reason, StopReason.BEHAVIOR_STOP)
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(first.last_content, second.last_content)

    async def test_continue_decision_allows_the_next_provider_turn(self) -> None:
        calls: list[str] = []
        hook = OrderedHook(
            "continue",
            calls,
            BehaviorDecision.continue_run(),
        )
        model = SequenceModel(
            [tool_turn(), AssistantMessage(content="finished normally")]
        )
        agent = BaseAgent(
            model,
            agent_id="behavior-proceed",
            handlers=[EchoHandler()],
            behavior_hooks=[hook],
        )

        result = await agent.run(task="proceed")

        self.assertEqual(result.stop_reason, StopReason.TEXT_RESPONSE)
        self.assertEqual(result.turns, 2)
        self.assertEqual(len(model.contexts), 2)
        self.assertEqual(calls, ["continue"])

    async def test_terminal_text_response_does_not_invoke_after_turn(self) -> None:
        hook = RaisingHook()
        sink = RecordingSink()
        agent = BaseAgent(
            SequenceModel([AssistantMessage(content="already terminal")]),
            agent_id="behavior-terminal",
            behavior_hooks=[hook],
            event_sink=sink,
        )

        result = await agent.run(task="finish normally")

        self.assertEqual(result.stop_reason, StopReason.TEXT_RESPONSE)
        self.assertEqual(
            sum(isinstance(event.payload, MessageCompleted) for event in sink.events),
            1,
        )

    async def test_hook_exception_and_invalid_decision_fail_the_run(self) -> None:
        for hook, expected in [
            (RaisingHook(), "hook exploded"),
            (InvalidHook(), "BehaviorDecision or None"),
        ]:
            with self.subTest(hook=type(hook).__name__):
                agent = BaseAgent(
                    SequenceModel([tool_turn()]),
                    agent_id=f"behavior-{type(hook).__name__}",
                    handlers=[EchoHandler()],
                    behavior_hooks=[hook],
                )

                result = await agent.run(task="fail in hook")

                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertEqual(result.stop_reason, StopReason.RUNTIME_ERROR)
                self.assertIn(expected, result.error or "")

    async def test_abort_interrupts_slow_hook_and_idle_waits_for_it(self) -> None:
        hook = BlockingHook()
        sink = RecordingSink()
        agent = BaseAgent(
            SequenceModel([tool_turn()]),
            agent_id="behavior-cancel",
            handlers=[EchoHandler()],
            behavior_hooks=[hook],
            event_sink=sink,
        )
        run = asyncio.create_task(agent.run(task="block in hook"))
        await hook.started.wait()
        idle = asyncio.create_task(agent.wait_for_idle())
        await asyncio.sleep(0)

        self.assertFalse(run.done())
        self.assertFalse(idle.done())
        self.assertTrue(agent.abort("cancel slow behavior"))
        result = await run
        await idle

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertTrue(hook.interrupted.is_set())
        self.assertIsInstance(sink.events[-1].payload, AgentFinished)

    async def test_continue_run_uses_the_same_behavior_hooks(self) -> None:
        hook = StopHook()
        model = SequenceModel(
            [AssistantMessage(content="initial"), tool_turn("continue-tool")]
        )
        agent = BaseAgent(
            model,
            agent_id="behavior-continue",
            handlers=[EchoHandler()],
            behavior_hooks=[hook],
        )
        await agent.run(task="initial")

        result = await agent.continue_run()

        self.assertEqual(result.stop_reason, StopReason.BEHAVIOR_STOP)
        self.assertIsNone(hook.snapshots[-1].task)


if __name__ == "__main__":
    unittest.main()
