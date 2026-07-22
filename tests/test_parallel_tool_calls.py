import asyncio
import json
import unittest
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
    RuntimePolicy,
    SessionRecorder,
    StepOutcome,
    StopReason,
    ToolCompleted,
    ToolControl,
    ToolDefinitionError,
    ToolEffect,
    ToolProgressed,
    ToolProgressReporter,
    ToolProgressUpdate,
    ToolStarted,
)


def tool_definition(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Run {name}.",
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
    }


READ_TOOL = tool_definition("read")
WRITE_TOOL = tool_definition("write")


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


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class ControlledHandler(MethodToolHandler):
    def __init__(
        self,
        *,
        read_only: bool = True,
        failures: set[str] | None = None,
        controls: dict[str, ToolControl] | None = None,
    ) -> None:
        effects = {"read": ToolEffect.READ_ONLY} if read_only else None
        super().__init__((READ_TOOL, WRITE_TOOL), tool_effects=effects)
        self.started: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}
        self.finished: dict[str, asyncio.Event] = {}
        self.failures = failures or set()
        self.controls = controls or {}
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    def started_for(self, label: str) -> asyncio.Event:
        return self.started.setdefault(label, asyncio.Event())

    def release_for(self, label: str) -> asyncio.Event:
        return self.release.setdefault(label, asyncio.Event())

    def finished_for(self, label: str) -> asyncio.Event:
        return self.finished.setdefault(label, asyncio.Event())

    async def do_read(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        return await self._execute(arguments, progress=progress)

    async def do_write(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        return await self._execute(arguments, progress=progress)

    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        progress: ToolProgressReporter | None,
    ) -> StepOutcome:
        label = str(arguments["label"])
        self.calls.append(label)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started_for(label).set()
        if progress is not None:
            await progress.report(ToolProgressUpdate(f"started {label}"))
        try:
            await self.release_for(label).wait()
            if label in self.failures:
                raise RuntimeError(f"failed {label}")
            return StepOutcome(
                {"label": label},
                control=self.controls.get(label, ToolControl.CONTINUE),
            )
        finally:
            self.active -= 1
            self.finished_for(label).set()


def call(call_id: str, name: str, label: str) -> ModelToolCall:
    return ModelToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps({"label": label}),
    )


def tool_payloads(sink: RecordingSink, payload_type: type[Any]) -> list[Any]:
    return [
        event.payload
        for event in sink.events
        if isinstance(event.payload, payload_type)
    ]


class ParallelToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_policy_keeps_declared_read_only_calls_sequential(
        self,
    ) -> None:
        handler = ControlledHandler()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=(
                            call("one", "read", "one"),
                            call("two", "read", "two"),
                        )
                    ),
                    AssistantMessage(content="done"),
                ]
            ),
            agent_id="parallel-default-off",
            handlers=[handler],
        )

        run = asyncio.create_task(agent.run(task="read twice"))
        await handler.started_for("one").wait()
        await asyncio.sleep(0)
        self.assertFalse(handler.started_for("two").is_set())
        handler.release_for("one").set()
        await handler.started_for("two").wait()
        handler.release_for("two").set()
        result = await run

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(handler.max_active, 1)

    async def test_read_only_calls_run_concurrently_but_commit_in_source_order(
        self,
    ) -> None:
        handler = ControlledHandler()
        sink = RecordingSink()
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="parallel-order", storage=storage)
        calls = (
            call("slow", "read", "slow"),
            call("fast", "read", "fast"),
        )
        agent = BaseAgent(
            SequenceModel(
                [AssistantMessage(tool_calls=calls), AssistantMessage(content="done")]
            ),
            agent_id="parallel-order",
            handlers=[handler],
            runtime_policy=RuntimePolicy(parallel_tool_calls=True),
            event_sink=CompositeAgentEventSink([recorder, sink]),
        )

        run = asyncio.create_task(agent.run(task="read concurrently"))
        await asyncio.gather(
            handler.started_for("slow").wait(),
            handler.started_for("fast").wait(),
        )
        handler.release_for("fast").set()
        await handler.finished_for("fast").wait()
        self.assertFalse(handler.finished_for("slow").is_set())
        handler.release_for("slow").set()
        result = await run
        session = await recorder.load()

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(handler.max_active, 2)
        self.assertEqual(
            [payload.tool_call.id for payload in tool_payloads(sink, ToolStarted)],
            ["slow", "fast"],
        )
        self.assertEqual(
            [payload.tool_call.id for payload in tool_payloads(sink, ToolCompleted)],
            ["slow", "fast"],
        )
        progressed = tool_payloads(sink, ToolProgressed)
        self.assertCountEqual(
            [payload.tool_call.id for payload in progressed],
            ["slow", "fast"],
        )
        first_completion = next(
            index
            for index, event in enumerate(sink.events)
            if isinstance(event.payload, ToolCompleted)
        )
        self.assertTrue(
            all(
                index < first_completion
                for index, event in enumerate(sink.events)
                if isinstance(event.payload, ToolProgressed)
            )
        )
        self.assertEqual(
            [
                message["tool_call_id"]
                for message in agent.messages
                if message["role"] == "tool"
            ],
            ["slow", "fast"],
        )
        assert session is not None
        self.assertEqual(
            [
                message["tool_call_id"]
                for message in session.messages
                if message["role"] == "tool"
            ],
            ["slow", "fast"],
        )

    async def test_side_effecting_call_is_a_barrier_between_read_batches(self) -> None:
        handler = ControlledHandler()
        calls = (
            call("r1", "read", "r1"),
            call("r2", "read", "r2"),
            call("w", "write", "w"),
            call("r3", "read", "r3"),
            call("r4", "read", "r4"),
        )
        agent = BaseAgent(
            SequenceModel(
                [AssistantMessage(tool_calls=calls), AssistantMessage(content="done")]
            ),
            agent_id="parallel-barrier",
            handlers=[handler],
            runtime_policy=RuntimePolicy(parallel_tool_calls=True),
        )

        run = asyncio.create_task(agent.run(task="mixed tools"))
        await asyncio.gather(
            handler.started_for("r1").wait(),
            handler.started_for("r2").wait(),
        )
        self.assertFalse(handler.started_for("w").is_set())
        handler.release_for("r1").set()
        handler.release_for("r2").set()
        await handler.started_for("w").wait()
        self.assertFalse(handler.started_for("r3").is_set())
        handler.release_for("w").set()
        await asyncio.gather(
            handler.started_for("r3").wait(),
            handler.started_for("r4").wait(),
        )
        handler.release_for("r3").set()
        handler.release_for("r4").set()
        result = await run

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(handler.calls, ["r1", "r2", "w", "r3", "r4"])
        self.assertEqual(handler.max_active, 2)

    async def test_parallel_limit_bounds_each_read_batch(self) -> None:
        handler = ControlledHandler()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=tuple(
                            call(label, "read", label)
                            for label in ("one", "two", "three")
                        )
                    ),
                    AssistantMessage(content="done"),
                ]
            ),
            agent_id="parallel-limit",
            handlers=[handler],
            runtime_policy=RuntimePolicy(
                parallel_tool_calls=True,
                max_parallel_tool_calls=2,
            ),
        )

        run = asyncio.create_task(agent.run(task="bounded reads"))
        await asyncio.gather(
            handler.started_for("one").wait(),
            handler.started_for("two").wait(),
        )
        self.assertFalse(handler.started_for("three").is_set())
        handler.release_for("one").set()
        handler.release_for("two").set()
        await handler.started_for("three").wait()
        handler.release_for("three").set()
        await run

        self.assertEqual(handler.max_active, 2)

    async def test_parallel_error_becomes_ordered_tool_result_and_loop_continues(
        self,
    ) -> None:
        handler = ControlledHandler(failures={"bad"})
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=(
                            call("bad", "read", "bad"),
                            call("good", "read", "good"),
                        )
                    ),
                    AssistantMessage(content="recovered"),
                ]
            ),
            agent_id="parallel-error",
            handlers=[handler],
            runtime_policy=RuntimePolicy(parallel_tool_calls=True),
        )

        run = asyncio.create_task(agent.run(task="recover"))
        await asyncio.gather(
            handler.started_for("bad").wait(),
            handler.started_for("good").wait(),
        )
        handler.release_for("good").set()
        handler.release_for("bad").set()
        result = await run
        tool_messages = [
            message for message in agent.messages if message["role"] == "tool"
        ]

        self.assertEqual(result.output, "recovered")
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["bad", "good"],
        )
        self.assertIn("failed bad", tool_messages[0]["content"])

    async def test_terminal_read_batch_settles_peers_before_stopping(self) -> None:
        handler = ControlledHandler(
            controls={
                "finish": ToolControl.COMPLETE,
                "peer": ToolControl.REJECT,
            }
        )
        sink = RecordingSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=(
                            call("finish", "read", "finish"),
                            call("peer", "read", "peer"),
                            call("late", "read", "late"),
                            call("write", "write", "write"),
                        )
                    )
                ]
            ),
            agent_id="parallel-terminal",
            handlers=[handler],
            runtime_policy=RuntimePolicy(
                parallel_tool_calls=True,
                max_parallel_tool_calls=2,
            ),
            event_sink=sink,
        )

        run = asyncio.create_task(agent.run(task="finish safely"))
        await asyncio.gather(
            handler.started_for("finish").wait(),
            handler.started_for("peer").wait(),
        )
        handler.release_for("peer").set()
        handler.release_for("finish").set()
        result = await run

        self.assertEqual(result.stop_reason, StopReason.TOOL_COMPLETION)
        self.assertEqual(json.loads(result.output or "{}"), {"label": "finish"})
        self.assertEqual(handler.calls, ["finish", "peer"])
        self.assertEqual(
            [payload.tool_call.id for payload in tool_payloads(sink, ToolCompleted)],
            ["finish", "peer", "late"],
        )
        self.assertTrue(tool_payloads(sink, ToolCompleted)[-1].result.cancelled)

    async def test_abort_settles_parallel_and_pending_calls_in_source_order(
        self,
    ) -> None:
        handler = ControlledHandler()
        sink = RecordingSink()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=(
                            call("one", "read", "one"),
                            call("two", "read", "two"),
                            call("write", "write", "write"),
                        )
                    )
                ]
            ),
            agent_id="parallel-cancel",
            handlers=[handler],
            runtime_policy=RuntimePolicy(parallel_tool_calls=True),
            event_sink=sink,
        )

        run = asyncio.create_task(agent.run(task="cancel reads"))
        await asyncio.gather(
            handler.started_for("one").wait(),
            handler.started_for("two").wait(),
        )
        self.assertTrue(agent.abort("stop parallel calls"))
        result = await run

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertEqual(handler.calls, ["one", "two"])
        self.assertEqual(
            [payload.tool_call.id for payload in tool_payloads(sink, ToolStarted)],
            ["one", "two", "write"],
        )
        completed = tool_payloads(sink, ToolCompleted)
        self.assertEqual(
            [payload.tool_call.id for payload in completed],
            ["one", "two", "write"],
        )
        self.assertTrue(all(payload.result.cancelled for payload in completed))
        self.assertEqual(
            [
                message["tool_call_id"]
                for message in agent.messages
                if message["role"] == "tool"
            ],
            ["one", "two", "write"],
        )

    async def test_repetition_guard_falls_back_to_ordered_execution(self) -> None:
        handler = ControlledHandler()
        repeated_calls = (
            call("one", "read", "same"),
            call("two", "read", "same"),
        )
        agent = BaseAgent(
            SequenceModel([AssistantMessage(tool_calls=repeated_calls)]),
            agent_id="parallel-repeat",
            handlers=[handler],
            runtime_policy=RuntimePolicy(
                max_repeated_tool_calls=2,
                parallel_tool_calls=True,
            ),
        )

        run = asyncio.create_task(agent.run(task="repeat"))
        await handler.started_for("same").wait()
        handler.release_for("same").set()
        result = await run

        self.assertEqual(result.stop_reason, StopReason.REPEATED_TOOL_CALL)
        self.assertEqual(handler.max_active, 1)

    def test_effect_and_parallel_limit_validation(self) -> None:
        with self.assertRaises(ToolDefinitionError):
            MethodToolHandler(
                (READ_TOOL,),
                tool_effects={"missing": ToolEffect.READ_ONLY},
            )
        with self.assertRaises(ToolDefinitionError):
            MethodToolHandler(
                (READ_TOOL,),
                tool_effects={"read": "read_only"},  # type: ignore[dict-item]
            )
        with self.assertRaises(ValueError):
            RuntimePolicy(max_parallel_tool_calls=0)
        with self.assertRaises(TypeError):
            RuntimePolicy(parallel_tool_calls="yes")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RuntimePolicy(max_parallel_tool_calls=True)


if __name__ == "__main__":
    unittest.main()
