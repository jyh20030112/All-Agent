from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    CancellationToken,
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
    ToolSemantics,
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


def lookup_definition() -> ToolDefinition:
    return ToolDefinition(
        name="lookup",
        description="Look up one value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        semantics=ToolSemantics.read_only(),
    )


def lookup_call(call_id: str = "lookup-1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="lookup",
        arguments={"key": "answer"},
    )


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
                "model_text_delta",
                "assistant_message",
                "turn_completed",
                "run_finished",
            ],
        )

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
