from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

from ejagent._trajectory import (
    CheckpointEvaluation,
    CheckpointEvaluationRequest,
    CheckpointSignal,
    CheckpointTrigger,
    EnvironmentFact,
    OnlineTrajectoryMonitor,
    TrajectoryContextBuffer,
    TrajectoryContextPipeline,
)
from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    CancellationToken,
    ContextRequest,
    FailureCode,
    ModelCallError,
    ModelPort,
    ModelProtocolError,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelUsage,
    RunIntent,
    RunLimits,
    RunPhase,
    RunSpec,
    RunStatus,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolControl,
    ToolDefinition,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutor,
    ToolProtocolError,
    ToolResultMessage,
    TransientInstruction,
    UserMessage,
    thaw_json_value,
)
from ejagent.kernel import RuntimeKernel


def fixed_clock() -> datetime:
    return datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


def task_spec(
    *,
    run_id: str = "run-1",
    base_revision: int = 4,
    task: str = "new task",
    limits: RunLimits | None = None,
) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        base_revision=base_revision,
        intent=RunIntent.TASK,
        task=task,
        messages=(SystemMessage("You are a test agent."),),
        limits=limits or RunLimits(),
        configuration_revision="config-1",
    )


class ScriptedModel(ModelPort):
    def __init__(
        self,
        responses: Sequence[
            tuple[AssistantMessage, ModelUsage | None] | ModelCallError
        ],
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
        message, usage = response
        if message.content:
            yield ModelTextDelta(message.content)
        yield ModelResponseCompleted(message, usage)


class IncompleteModel(ModelPort):
    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        yield ModelTextDelta("partial")


class BlockingModel(ModelPort):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.started.set()
        await self.release.wait()
        yield ModelResponseCompleted(AssistantMessage(content="late"))


class RecordingTools(ToolExecutor):
    def __init__(
        self,
        definitions: Sequence[ToolDefinition] = (),
        *,
        results: dict[str, ToolExecutionResult | ToolExecutionError] | None = None,
    ) -> None:
        self._definitions = tuple(definitions)
        self.results = results or {}
        self.calls: list[ToolCall] = []

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return self._definitions

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        self.calls.append(call)
        result = self.results.get(call.id, ToolExecutionResult({"ok": True}))
        if isinstance(result, ToolExecutionError):
            raise result
        return result


class ConcurrentTools(ToolExecutor):
    def __init__(self) -> None:
        self.started = {
            "lookup-1": asyncio.Event(),
            "lookup-2": asyncio.Event(),
        }
        self.release = {
            "lookup-1": asyncio.Event(),
            "lookup-2": asyncio.Event(),
        }
        self.completed = {
            "lookup-1": asyncio.Event(),
            "lookup-2": asyncio.Event(),
        }
        self.completion_order: list[str] = []

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return (lookup_definition(),)

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        self.started[call.id].set()
        await self.release[call.id].wait()
        self.completion_order.append(call.id)
        self.completed[call.id].set()
        return ToolExecutionResult({"call_id": call.id})


def lookup_definition() -> ToolDefinition:
    return ToolDefinition(
        name="lookup",
        description="Look up one value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    )


def lookup_call(call_id: str = "lookup-1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="lookup",
        arguments={"key": "answer"},
    )


class RuntimeTrajectoryEvaluator:
    def __init__(self) -> None:
        self.requests: list[CheckpointEvaluationRequest] = []

    async def evaluate(
        self,
        request: CheckpointEvaluationRequest,
        *,
        cancellation: CancellationToken,
    ) -> CheckpointEvaluation:
        cancellation.raise_if_cancelled()
        self.requests.append(request)
        trigger = request.signal.trigger
        state = "s0" if trigger is CheckpointTrigger.BASELINE else "s1"
        fact = EnvironmentFact(
            fact_id=f"{request.checkpoint_id}:state",
            subject="fixture/target",
            predicate="state",
            value=state,
            scope=("R1",),
            source="runtime-test-verifier",
            observed_at=fixed_clock(),
            checkpoint_id=request.checkpoint_id,
            evidence_ref=f"fixture://{trigger.value}",
            freshness="valid for this capture",
            authority="fixture target State only",
        )
        return CheckpointEvaluation(
            projection_version="runtime-test-v1",
            state_fingerprint=state,
            environment_facts={"state": state},
            requirements={"R1": False},
            constraints={},
            new_evidence=(trigger.value,),
            facts=(fact,),
            fact_capture_complete=True,
        )


class FailingTrajectoryMonitor:
    def __init__(self) -> None:
        self.signals: list[CheckpointSignal] = []
        self.closed_runs: list[str] = []

    async def capture(
        self,
        signal: CheckpointSignal,
        *,
        cancellation: CancellationToken,
    ) -> object:
        cancellation.raise_if_cancelled()
        self.signals.append(signal)
        raise RuntimeError("verifier unavailable")

    def close_run(self, run_id: str) -> object:
        self.closed_runs.append(run_id)
        return ()


class RuntimeKernelTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_run_returns_delta_without_mutating_spec(self) -> None:
        spec = task_spec()
        original_messages = spec.messages
        model = ScriptedModel(
            [
                (
                    AssistantMessage(content="done"),
                    ModelUsage(
                        input_tokens=10,
                        output_tokens=2,
                        total_tokens=12,
                    ),
                )
            ]
        )
        kernel = RuntimeKernel(
            model=model,
            tools=RecordingTools(),
            clock=fixed_clock,
        )

        outcome = await kernel.run(spec)

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(outcome.result.stop_reason, StopReason.TEXT_RESPONSE)
        self.assertEqual(outcome.result.output, "done")
        self.assertEqual(outcome.result.usage.total_tokens, 12)
        self.assertEqual(
            outcome.delta.messages,
            (UserMessage("new task"), AssistantMessage(content="done")),
        )
        self.assertEqual(outcome.delta.base_revision, 4)
        self.assertEqual(spec.messages, original_messages)
        self.assertEqual(
            model.requests[0].messages,
            (*original_messages, UserMessage("new task")),
        )
        self.assertEqual(
            [record.kind for record in outcome.audit_records],
            [
                "run_started",
                "turn_started",
                "context_built",
                "model_text_delta",
                "assistant_message",
                "turn_completed",
                "run_finished",
            ],
        )

    async def test_opt_in_trajectory_captures_runtime_boundaries_and_context(
        self,
    ) -> None:
        calls = (lookup_call("lookup-1"), lookup_call("lookup-2"))
        model = ScriptedModel(
            [
                (AssistantMessage(tool_calls=calls), None),
                (AssistantMessage(content="done"), None),
            ]
        )
        evaluator = RuntimeTrajectoryEvaluator()
        buffer = TrajectoryContextBuffer()
        monitor = OnlineTrajectoryMonitor(
            evaluator,
            update_sink=lambda update: buffer.publish(
                update.to_context_frame(
                    goal="new task",
                    next_turn=update.signal.turn + 1,
                )
            ),
            run_close_sink=lambda run_id, _checkpoints: buffer.close_run(run_id),
        )
        pipeline = TrajectoryContextPipeline(source=buffer)
        kernel = RuntimeKernel(
            model=model,
            tools=RecordingTools((lookup_definition(),)),
            context=pipeline,
            trajectory=monitor,
            clock=fixed_clock,
        )
        await pipeline.start()
        try:
            outcome = await kernel.run(task_spec())
        finally:
            await pipeline.shutdown()

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(
            [request.signal.trigger for request in evaluator.requests],
            [
                CheckpointTrigger.BASELINE,
                CheckpointTrigger.TOOL_BATCH_COMPLETED,
                CheckpointTrigger.COMPLETION_PROPOSED,
            ],
        )
        tool_signal = evaluator.requests[1].signal
        self.assertEqual(tool_signal.turn, 1)
        self.assertEqual(tool_signal.cumulative_cost.actor_actions, 2)
        self.assertEqual(tool_signal.cumulative_cost.model_requests, 1)
        self.assertEqual(tool_signal.causal_batch_id, "turn-1:tool-batch")
        self.assertEqual(
            tuple(action.action_id for action in tool_signal.causal_actions),
            ("lookup-1", "lookup-2"),
        )
        self.assertTrue(
            all(
                "answer" not in action.signature
                for action in tool_signal.causal_actions
            )
        )
        trajectory_audit = [
            record
            for record in outcome.audit_records
            if record.kind == "trajectory_checkpointed"
        ]
        self.assertEqual(len(trajectory_audit), 3)
        self.assertFalse(trajectory_audit[-1].payload["completion_allowed"])
        for request, checkpoint_id in zip(
            model.requests,
            ("run-1:cp0", "run-1:cp1"),
            strict=True,
        ):
            projected = tuple(
                message
                for message in request.messages
                if isinstance(message, TransientInstruction)
            )
            self.assertEqual(len(projected), 1)
            self.assertIn(f'"checkpoint":"{checkpoint_id}"', projected[0].content)
        self.assertEqual(monitor.checkpoints("run-1"), ())
        self.assertIsNone(
            buffer(
                ContextRequest(
                    run_id="run-1",
                    source_revision=0,
                    turn=3,
                    committed_messages=(SystemMessage("stable"),),
                )
            )
        )

    async def test_trajectory_failure_is_audited_and_does_not_change_result(
        self,
    ) -> None:
        monitor = FailingTrajectoryMonitor()
        outcome = await RuntimeKernel(
            model=ScriptedModel([(AssistantMessage(content="done"), None)]),
            tools=RecordingTools(),
            trajectory=monitor,
            clock=fixed_clock,
        ).run(task_spec())

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(outcome.result.output, "done")
        self.assertEqual(
            [signal.trigger for signal in monitor.signals],
            [CheckpointTrigger.BASELINE],
        )
        self.assertEqual(monitor.closed_runs, ["run-1"])
        failure_records = tuple(
            record
            for record in outcome.audit_records
            if record.kind == "trajectory_capture_failed"
        )
        self.assertEqual(len(failure_records), 1)
        self.assertEqual(failure_records[0].payload["error_type"], "RuntimeError")

    async def test_trajectory_state_closes_when_runtime_protocol_error_escapes(
        self,
    ) -> None:
        monitor = OnlineTrajectoryMonitor(RuntimeTrajectoryEvaluator())
        kernel = RuntimeKernel(
            model=IncompleteModel(),
            tools=RecordingTools(),
            trajectory=monitor,
            clock=fixed_clock,
        )

        with self.assertRaises(ModelProtocolError):
            await kernel.run(task_spec())

        self.assertEqual(monitor.checkpoints("run-1"), ())

    async def test_tool_result_is_committed_before_the_next_model_request(
        self,
    ) -> None:
        call = lookup_call()
        model = ScriptedModel(
            [
                (AssistantMessage(tool_calls=(call,)), None),
                (AssistantMessage(content="answer is 42"), None),
            ]
        )
        tools = RecordingTools(
            (lookup_definition(),),
            results={call.id: ToolExecutionResult({"value": 42})},
        )
        kernel = RuntimeKernel(model=model, tools=tools, clock=fixed_clock)

        outcome = await kernel.run(task_spec())

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(len(model.requests), 2)
        tool_message = model.requests[1].messages[-1]
        self.assertIsInstance(tool_message, ToolResultMessage)
        assert isinstance(tool_message, ToolResultMessage)
        self.assertEqual(thaw_json_value(tool_message.result), {"value": 42})
        self.assertEqual(
            outcome.delta.messages,
            (
                UserMessage("new task"),
                AssistantMessage(tool_calls=(call,)),
                tool_message,
                AssistantMessage(content="answer is 42"),
            ),
        )

    async def test_tool_batch_runs_concurrently_and_commits_in_source_order(
        self,
    ) -> None:
        calls = (lookup_call("lookup-1"), lookup_call("lookup-2"))
        model = ScriptedModel(
            [
                (AssistantMessage(tool_calls=calls), None),
                (AssistantMessage(content="done"), None),
            ]
        )
        tools = ConcurrentTools()
        running = asyncio.create_task(
            RuntimeKernel(model=model, tools=tools, clock=fixed_clock).run(task_spec())
        )

        await asyncio.wait_for(tools.started["lookup-1"].wait(), timeout=0.2)
        await asyncio.wait_for(tools.started["lookup-2"].wait(), timeout=0.2)
        tools.release["lookup-2"].set()
        await tools.completed["lookup-2"].wait()
        tools.release["lookup-1"].set()
        outcome = await running

        self.assertEqual(tools.completion_order, ["lookup-2", "lookup-1"])
        results = tuple(
            message
            for message in outcome.delta.messages
            if isinstance(message, ToolResultMessage)
        )
        self.assertEqual(
            tuple(message.tool_call_id for message in results),
            ("lookup-1", "lookup-2"),
        )

    async def test_completing_tool_terminates_without_another_model_request(
        self,
    ) -> None:
        call = lookup_call()
        model = ScriptedModel([(AssistantMessage(tool_calls=(call,)), None)])
        tools = RecordingTools(
            (lookup_definition(),),
            results={
                call.id: ToolExecutionResult(
                    {"value": 42},
                    control=ToolControl.COMPLETE,
                    output="finished",
                )
            },
        )

        outcome = await RuntimeKernel(
            model=model,
            tools=tools,
            clock=fixed_clock,
        ).run(task_spec())

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(outcome.result.stop_reason, StopReason.TOOL_COMPLETION)
        self.assertEqual(outcome.result.output, "finished")
        self.assertEqual(len(model.requests), 1)

    async def test_unknown_tool_becomes_an_error_result_for_the_model(self) -> None:
        call = ToolCall(id="missing-1", name="missing")
        model = ScriptedModel(
            [
                (AssistantMessage(tool_calls=(call,)), None),
                (AssistantMessage(content="recovered"), None),
            ]
        )
        tools = RecordingTools()

        outcome = await RuntimeKernel(
            model=model,
            tools=tools,
            clock=fixed_clock,
        ).run(task_spec())

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(tools.calls, [])
        result = model.requests[1].messages[-1]
        self.assertIsInstance(result, ToolResultMessage)
        assert isinstance(result, ToolResultMessage)
        self.assertTrue(result.is_error)

    async def test_model_operational_error_returns_structured_failure(self) -> None:
        model = ScriptedModel(
            [
                ModelCallError(
                    FailureCode.TIMEOUT,
                    "provider timed out",
                    retryable=True,
                )
            ]
        )

        outcome = await RuntimeKernel(
            model=model,
            tools=RecordingTools(),
            clock=fixed_clock,
        ).run(task_spec())

        self.assertEqual(outcome.result.status, RunStatus.FAILED)
        self.assertEqual(outcome.failure.phase, RunPhase.MODEL)
        self.assertEqual(outcome.failure.code, FailureCode.TIMEOUT)
        self.assertTrue(outcome.failure.retryable)
        self.assertEqual(outcome.delta.messages, (UserMessage("new task"),))

    async def test_tool_infrastructure_error_returns_structured_failure(self) -> None:
        call = lookup_call()
        model = ScriptedModel([(AssistantMessage(tool_calls=(call,)), None)])
        tools = RecordingTools(
            (lookup_definition(),),
            results={
                call.id: ToolExecutionError("executor unavailable", retryable=True)
            },
        )

        outcome = await RuntimeKernel(
            model=model,
            tools=tools,
            clock=fixed_clock,
        ).run(task_spec())

        self.assertEqual(outcome.result.status, RunStatus.FAILED)
        self.assertEqual(outcome.failure.phase, RunPhase.TOOL)
        self.assertTrue(outcome.failure.retryable)
        self.assertEqual(
            outcome.delta.messages,
            (UserMessage("new task"), AssistantMessage(tool_calls=(call,))),
        )

    async def test_incomplete_model_stream_raises_protocol_error(self) -> None:
        kernel = RuntimeKernel(
            model=IncompleteModel(),
            tools=RecordingTools(),
            clock=fixed_clock,
        )

        with self.assertRaisesRegex(ModelProtocolError, "without completion"):
            await kernel.run(task_spec())

    async def test_cancellation_returns_partial_auditable_outcome(self) -> None:
        model = BlockingModel()
        source = CancellationSource()
        kernel = RuntimeKernel(
            model=model,
            tools=RecordingTools(),
            clock=fixed_clock,
        )

        task = asyncio.create_task(kernel.run(task_spec(), cancellation=source.token))
        await model.started.wait()
        source.cancel("stop now")
        outcome = await task

        self.assertEqual(outcome.result.status, RunStatus.CANCELLED)
        self.assertEqual(outcome.result.stop_reason, StopReason.EXTERNAL_ABORT)
        self.assertEqual(outcome.result.output, "stop now")
        self.assertEqual(outcome.delta.messages, (UserMessage("new task"),))

    async def test_max_turns_returns_failure_with_partial_delta(self) -> None:
        call = lookup_call()
        model = ScriptedModel([(AssistantMessage(tool_calls=(call,)), None)])
        kernel = RuntimeKernel(
            model=model,
            tools=RecordingTools((lookup_definition(),)),
            clock=fixed_clock,
        )

        outcome = await kernel.run(task_spec(limits=RunLimits(max_turns=1)))

        self.assertEqual(outcome.result.status, RunStatus.FAILED)
        self.assertEqual(outcome.result.stop_reason, StopReason.MAX_STEPS)
        self.assertEqual(outcome.failure.code, FailureCode.BUDGET_EXCEEDED)

    async def test_repeated_tool_guard_rejects_before_third_execution(self) -> None:
        calls = tuple(lookup_call(f"lookup-{index}") for index in range(3))
        model = ScriptedModel(
            [(AssistantMessage(tool_calls=(call,)), None) for call in calls]
        )
        tools = RecordingTools((lookup_definition(),))
        kernel = RuntimeKernel(model=model, tools=tools, clock=fixed_clock)

        outcome = await kernel.run(task_spec())

        self.assertEqual(outcome.result.stop_reason, StopReason.REPEATED_TOOL_CALL)
        self.assertEqual(len(tools.calls), 2)

    async def test_missing_usage_stops_before_second_budgeted_request(self) -> None:
        call = lookup_call()
        model = ScriptedModel([(AssistantMessage(tool_calls=(call,)), None)])
        kernel = RuntimeKernel(
            model=model,
            tools=RecordingTools((lookup_definition(),)),
            clock=fixed_clock,
        )

        outcome = await kernel.run(task_spec(limits=RunLimits(max_tokens=10)))

        self.assertEqual(outcome.result.stop_reason, StopReason.USAGE_UNAVAILABLE)
        self.assertEqual(len(model.requests), 1)

    async def test_duplicate_tool_definitions_are_protocol_error(self) -> None:
        definition = lookup_definition()
        kernel = RuntimeKernel(
            model=ScriptedModel([(AssistantMessage(content="unused"), None)]),
            tools=RecordingTools((definition, definition)),
            clock=fixed_clock,
        )

        with self.assertRaisesRegex(ToolProtocolError, "duplicate"):
            await kernel.run(task_spec())


if __name__ == "__main__":
    unittest.main()
