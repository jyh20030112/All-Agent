import asyncio
import json
import unittest
from typing import Any

from ejagent import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    AgentStatus,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    CompositeAgentEventSink,
    McpToolHandler,
    MemorySessionStorage,
    MethodToolHandler,
    ModelAdapter,
    ModelToolCall,
    RunStatus,
    SessionRecorder,
    StepOutcome,
    StopReason,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
    TurnStarted,
)

WAIT_TOOL = {
    "type": "function",
    "function": {
        "name": "wait",
        "description": "Wait until the operation is cancelled.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class BlockingThenCompleteModel(ModelAdapter):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.calls = 0
        self.tokens: list[CancellationToken] = []

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        assert cancellation is not None
        self.tokens.append(cancellation)
        self.calls += 1
        if self.calls > 1:
            return AssistantMessage(content="reused")

        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.interrupted.set()


class SequenceModel(ModelAdapter):
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = list(responses)

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        return self.responses.pop(0)


class BlockingToolHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((WAIT_TOOL,))
        self.started = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.calls = 0
        self.token: CancellationToken | None = None

    async def do_wait(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        assert cancellation is not None
        self.calls += 1
        self.token = cancellation
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.interrupted.set()


class BlockingMcpManager:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.interrupted = asyncio.Event()

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "mcp__wait",
                    "description": "Wait in an MCP server.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.interrupted.set()


class SlowFinishSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.finish_started = asyncio.Event()
        self.release = asyncio.Event()

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)
        if isinstance(event.payload, AgentFinished):
            self.finish_started.set()
            await self.release.wait()


def payloads(sink: RecordingSink) -> list[object]:
    return [event.payload for event in sink.events]


class AgentCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_abort_interrupts_model_and_agent_can_be_reused(self) -> None:
        model = BlockingThenCompleteModel()
        sink = RecordingSink()
        agent = BaseAgent(model, agent_id="abort-model", event_sink=sink)

        self.assertFalse(agent.abort())
        first_run = asyncio.create_task(agent.run(task="block"))
        await model.started.wait()

        self.assertTrue(agent.abort("stopped by user"))
        self.assertFalse(agent.abort("duplicate"))
        first_result = await first_run
        await agent.wait_for_idle()

        self.assertTrue(model.interrupted.is_set())
        self.assertEqual(first_result.status, RunStatus.CANCELLED)
        self.assertEqual(first_result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertEqual(first_result.error, "stopped by user")
        self.assertEqual(agent.state.status, AgentStatus.CANCELLED)
        self.assertEqual(
            [type(payload) for payload in payloads(sink)],
            [
                AgentStarted,
                TurnStarted,
                TurnCompleted,
                AgentFinished,
            ],
        )

        second_result = await agent.run(task="run again")

        self.assertEqual(second_result.status, RunStatus.COMPLETED)
        self.assertEqual(second_result.output, "reused")
        self.assertEqual(model.calls, 2)
        self.assertIsNot(model.tokens[0], model.tokens[1])
        self.assertTrue(model.tokens[0].cancelled)
        self.assertFalse(model.tokens[1].cancelled)
        self.assertFalse(agent.abort())

    async def test_tool_abort_closes_all_tool_calls_and_session_run(self) -> None:
        first_call = ModelToolCall(
            id="call-1",
            name="wait",
            arguments="{}",
        )
        second_call = ModelToolCall(
            id="call-2",
            name="wait",
            arguments="{}",
        )
        handler = BlockingToolHandler()
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="cancel-tool", storage=storage)
        observer = RecordingSink()
        agent = BaseAgent(
            SequenceModel([AssistantMessage(tool_calls=(first_call, second_call))]),
            agent_id="abort-tool",
            handlers=[handler],
            event_sink=CompositeAgentEventSink([recorder, observer]),
        )

        run = asyncio.create_task(agent.run(task="use tools"))
        await handler.started.wait()
        self.assertTrue(agent.abort())
        result = await run
        session = await recorder.load()

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertEqual(handler.calls, 1)
        self.assertTrue(handler.interrupted.is_set())
        assert handler.token is not None
        self.assertTrue(handler.token.cancelled)

        tool_started = [
            payload
            for payload in payloads(observer)
            if isinstance(payload, ToolStarted)
        ]
        tool_completed = [
            payload
            for payload in payloads(observer)
            if isinstance(payload, ToolCompleted)
        ]
        self.assertEqual(len(tool_started), 2)
        self.assertEqual(len(tool_completed), 2)
        self.assertTrue(all(payload.result.cancelled for payload in tool_completed))
        self.assertEqual(
            [message["role"] for message in agent.messages[-3:]],
            ["assistant", "tool", "tool"],
        )
        for message in agent.messages[-2:]:
            self.assertEqual(
                json.loads(message["content"])["status"],
                "cancelled",
            )

        assert session is not None
        self.assertTrue(session.runs[0].finished)
        assert session.runs[0].result is not None
        self.assertEqual(session.runs[0].result.status, RunStatus.CANCELLED)
        self.assertEqual(
            [message["role"] for message in session.messages],
            ["user", "assistant", "tool", "tool"],
        )

    async def test_wait_for_idle_includes_agent_finished_sink(self) -> None:
        model = BlockingThenCompleteModel()
        sink = SlowFinishSink()
        agent = BaseAgent(model, agent_id="idle-settlement", event_sink=sink)
        run = asyncio.create_task(agent.run(task="block"))
        await model.started.wait()

        agent.abort()
        await sink.finish_started.wait()
        idle = asyncio.create_task(agent.wait_for_idle())
        await asyncio.sleep(0)

        self.assertFalse(idle.done())
        self.assertFalse(run.done())
        sink.release.set()
        await idle
        result = await run
        self.assertEqual(result.status, RunStatus.CANCELLED)

    async def test_mcp_only_agent_can_abort_an_active_tool_call(self) -> None:
        manager = BlockingMcpManager()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=(
                            ModelToolCall(
                                id="mcp-call",
                                name="mcp__wait",
                                arguments="{}",
                            ),
                        )
                    )
                ]
            ),
            agent_id="abort-mcp",
            handlers=[McpToolHandler(manager=manager)],
        )

        run = asyncio.create_task(agent.run(task="call MCP"))
        await manager.started.wait()
        agent.abort()
        result = await run

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertTrue(manager.interrupted.is_set())
        self.assertEqual(agent.messages[-1]["role"], "tool")

    async def test_caller_task_cancellation_still_emits_terminal_events(
        self,
    ) -> None:
        model = BlockingThenCompleteModel()
        sink = RecordingSink()
        agent = BaseAgent(model, agent_id="caller-cancel", event_sink=sink)
        run = asyncio.create_task(agent.run(task="block"))
        await model.started.wait()

        run.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run
        await agent.wait_for_idle()

        event_payloads = payloads(sink)
        self.assertIsInstance(event_payloads[-2], TurnCompleted)
        self.assertIsInstance(event_payloads[-1], AgentFinished)
        finish = event_payloads[-1]
        assert isinstance(finish, AgentFinished)
        self.assertEqual(finish.result.status, RunStatus.CANCELLED)
        self.assertEqual(finish.result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertEqual(agent.state.status, AgentStatus.CANCELLED)
        self.assertTrue(model.interrupted.is_set())


if __name__ == "__main__":
    unittest.main()
