import asyncio
import unittest
from dataclasses import FrozenInstanceError
from typing import Any

from ejagent import (
    AgentEvent,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    CompositeAgentEventSink,
    MemorySessionStorage,
    MethodToolHandler,
    ModelAdapter,
    ModelToolCall,
    RunStatus,
    SessionRecorder,
    StepOutcome,
    StopReason,
    ToolCallContext,
    ToolCompleted,
    ToolMiddleware,
    ToolNext,
    ToolProgressed,
    ToolProgressReporter,
    ToolProgressUpdate,
    ToolStarted,
)

WORK_TOOL = {
    "type": "function",
    "function": {
        "name": "work",
        "description": "Perform observable work.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


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


class ProgressHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((WORK_TOOL,))
        self.reporter: ToolProgressReporter | None = None

    async def do_work(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        assert progress is not None
        self.reporter = progress
        await progress.report(
            ToolProgressUpdate("scanning", {"completed": 1, "total": 2})
        )
        await progress.report(
            ToolProgressUpdate("finished", {"completed": 2, "total": 2})
        )
        return StepOutcome({"files": 2})


class LateProgressHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((WORK_TOOL,))
        self.release = asyncio.Event()
        self.late_report: asyncio.Task[None] | None = None

    async def do_work(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        assert progress is not None
        await progress.report(ToolProgressUpdate("accepted"))

        async def report_late() -> None:
            await self.release.wait()
            await progress.report(ToolProgressUpdate("too late"))

        self.late_report = asyncio.create_task(report_late())
        return StepOutcome({"status": "done"})


class BlockingProgressHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((WORK_TOOL,))
        self.started = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.late_report: asyncio.Task[None] | None = None

    async def do_work(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        assert progress is not None
        await progress.report(ToolProgressUpdate("waiting"))
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.interrupted.set()
            self.late_report = asyncio.create_task(
                progress.report(ToolProgressUpdate("interrupted"))
            )


class MiddlewareProgress(ToolMiddleware):
    async def __call__(
        self,
        context: ToolCallContext,
        call_next: ToolNext,
    ) -> StepOutcome:
        assert context.progress is not None
        await context.progress.report(ToolProgressUpdate("middleware"))
        return await call_next(context)


def tool_call() -> ModelToolCall:
    return ModelToolCall(id="work-1", name="work", arguments="{}")


def payloads(sink: RecordingSink) -> list[object]:
    return [event.payload for event in sink.events]


class ToolProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_is_ordered_and_correlated_to_tool_call(self) -> None:
        call = tool_call()
        sink = RecordingSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(tool_calls=(call,)),
                    AssistantMessage(content="done"),
                ]
            ),
            agent_id="progress-events",
            handlers=[ProgressHandler()],
            event_sink=sink,
        )

        result = await agent.run(task="work")

        tool_payloads = [
            payload
            for payload in payloads(sink)
            if isinstance(payload, (ToolStarted, ToolProgressed, ToolCompleted))
        ]
        self.assertEqual(
            [type(payload) for payload in tool_payloads],
            [ToolStarted, ToolProgressed, ToolProgressed, ToolCompleted],
        )
        progress = tool_payloads[1:3]
        self.assertEqual(
            [payload.update.message for payload in progress],
            ["scanning", "finished"],
        )
        self.assertTrue(all(payload.tool_call is call for payload in tool_payloads))
        self.assertEqual(
            [event.sequence for event in sink.events],
            list(range(1, len(sink.events) + 1)),
        )
        self.assertEqual(len({event.run_id for event in sink.events}), 1)
        self.assertEqual(result.status, RunStatus.COMPLETED)

    async def test_progress_is_not_persisted_in_session(self) -> None:
        call = tool_call()
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="progress", storage=storage)
        observer = RecordingSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(tool_calls=(call,)),
                    AssistantMessage(content="done"),
                ]
            ),
            agent_id="progress-session",
            handlers=[ProgressHandler()],
            event_sink=CompositeAgentEventSink([recorder, observer]),
        )

        await agent.run(task="work")
        session = await recorder.load()

        self.assertEqual(
            len(
                [
                    payload
                    for payload in payloads(observer)
                    if isinstance(payload, ToolProgressed)
                ]
            ),
            2,
        )
        assert session is not None
        self.assertEqual(
            [message["role"] for message in session.messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertNotIn("scanning", repr(session))
        self.assertNotIn("finished", repr(session))

    async def test_middleware_can_report_on_the_same_tool_call(self) -> None:
        sink = RecordingSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(tool_calls=(tool_call(),)),
                    AssistantMessage(content="done"),
                ]
            ),
            agent_id="middleware-progress",
            handlers=[ProgressHandler()],
            middlewares=[MiddlewareProgress()],
            event_sink=sink,
        )

        await agent.run(task="work")

        updates = [
            payload.update.message
            for payload in payloads(sink)
            if isinstance(payload, ToolProgressed)
        ]
        self.assertEqual(updates, ["middleware", "scanning", "finished"])

    async def test_late_progress_is_ignored_after_tool_completion(self) -> None:
        handler = LateProgressHandler()
        sink = RecordingSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(tool_calls=(tool_call(),)),
                    AssistantMessage(content="done"),
                ]
            ),
            agent_id="late-progress",
            handlers=[handler],
            event_sink=sink,
        )

        await agent.run(task="work")
        event_count = len(sink.events)
        handler.release.set()
        assert handler.late_report is not None
        await handler.late_report

        self.assertEqual(len(sink.events), event_count)
        updates = [
            payload.update.message
            for payload in payloads(sink)
            if isinstance(payload, ToolProgressed)
        ]
        self.assertEqual(updates, ["accepted"])

    async def test_abort_rejects_progress_after_cancellation(self) -> None:
        handler = BlockingProgressHandler()
        sink = RecordingSink()
        agent = BaseAgent(
            SequenceModel([AssistantMessage(tool_calls=(tool_call(),))]),
            agent_id="abort-progress",
            handlers=[handler],
            event_sink=sink,
        )

        run = asyncio.create_task(agent.run(task="work"))
        await handler.started.wait()
        agent.abort("stop progress")
        result = await run
        assert handler.late_report is not None
        await handler.late_report

        tool_payloads = [
            payload
            for payload in payloads(sink)
            if isinstance(payload, (ToolStarted, ToolProgressed, ToolCompleted))
        ]
        self.assertEqual(
            [type(payload) for payload in tool_payloads],
            [ToolStarted, ToolProgressed, ToolCompleted],
        )
        self.assertEqual(tool_payloads[1].update.message, "waiting")
        self.assertTrue(tool_payloads[-1].result.cancelled)
        self.assertTrue(handler.interrupted.is_set())
        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.stop_reason, StopReason.EXTERNAL_ABORT)

    async def test_progress_update_is_validated_and_immutable(self) -> None:
        with self.assertRaises(ValueError):
            ToolProgressUpdate("")

        update = ToolProgressUpdate("working")
        with self.assertRaises(FrozenInstanceError):
            update.message = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
