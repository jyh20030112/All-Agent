from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime

from ejagent.context import IdentityContextPipeline
from ejagent.contracts.context import (
    ContextBuildError,
    ContextPipeline,
    ContextProtocolError,
    ContextRequest,
    ContextView,
)
from ejagent.contracts.control import (
    CancellationSource,
    CancellationToken,
    ControlProtocolError,
    RunCancelledError,
    RunControlSource,
    SteeringInput,
)
from ejagent.contracts.json import (
    JsonObject,
    JsonValue,
    freeze_json_object,
)
from ejagent.contracts.messages import (
    AssistantMessage,
    ToolCall,
    TransientInstruction,
)
from ejagent.contracts.model import (
    ModelCallError,
    ModelPort,
    ModelProtocolError,
    ModelRequest,
    ModelResponseCompleted,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelUsage,
)
from ejagent.contracts.runs import (
    AuditRecord,
    FailureCode,
    RunFailure,
    RunOutcome,
    RunPhase,
    RunResult,
    RunSpec,
    RunStatus,
    StopReason,
)
from ejagent.contracts.tools import (
    ToolControl,
    ToolDefinition,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutor,
    ToolProtocolError,
)
from ejagent.contracts.usage import RunUsage
from ejagent.kernel._workspace import _RunWorkspace

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _UsageAccumulator:
    def __init__(self) -> None:
        self.request_count = 0
        self.reported_request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens: int | None = None
        self.cache_write_tokens: int | None = None
        self.reasoning_tokens: int | None = None

    def begin_request(self) -> None:
        self.request_count += 1

    def record(self, usage: ModelUsage | None) -> None:
        if usage is None:
            return
        previous_reports = self.reported_request_count
        self.reported_request_count += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens = self._add_optional(
            self.cache_read_tokens,
            usage.cache_read_tokens,
            previous_reports,
        )
        self.cache_write_tokens = self._add_optional(
            self.cache_write_tokens,
            usage.cache_write_tokens,
            previous_reports,
        )
        self.reasoning_tokens = self._add_optional(
            self.reasoning_tokens,
            usage.reasoning_tokens,
            previous_reports,
        )

    def snapshot(self) -> RunUsage:
        return RunUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            request_count=self.request_count,
            reported_request_count=self.reported_request_count,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens,
        )

    @staticmethod
    def _add_optional(
        current: int | None,
        value: int | None,
        previous_reports: int,
    ) -> int | None:
        if previous_reports == 0:
            return value
        if current is None or value is None:
            return None
        return current + value


class _AuditTrail:
    def __init__(self, run_id: str, clock: Clock) -> None:
        self.run_id = run_id
        self.clock = clock
        self.records: list[AuditRecord] = []

    def append(self, kind: str, payload: Mapping[str, JsonValue] | None = None) -> None:
        self.records.append(
            AuditRecord(
                run_id=self.run_id,
                sequence=len(self.records) + 1,
                kind=kind,
                occurred_at=self.clock(),
                payload=payload or {},
            )
        )


class RuntimeKernel:
    """Execute one deterministic Model-Tool Run over a private workspace."""

    def __init__(
        self,
        *,
        model: ModelPort,
        tools: ToolExecutor,
        context: ContextPipeline | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._context = context if context is not None else IdentityContextPipeline()
        self._clock = clock or _utc_now

    async def run(
        self,
        spec: RunSpec,
        *,
        cancellation: CancellationToken | None = None,
        controls: RunControlSource | None = None,
    ) -> RunOutcome:
        """Execute one Run without mutating the supplied RunSpec or Harness state."""

        if not isinstance(spec, RunSpec):
            raise TypeError("spec must be a RunSpec")
        token = cancellation or CancellationSource().token
        workspace = _RunWorkspace(spec)
        usage = _UsageAccumulator()
        audit = _AuditTrail(spec.run_id, self._clock)
        definitions = self._snapshot_tool_definitions()
        tool_names = {definition.name for definition in definitions}
        audit.append(
            "run_started",
            {
                "base_revision": spec.base_revision,
                "intent": spec.intent.value,
                "configuration_revision": spec.configuration_revision,
            },
        )

        try:
            for _ in range(spec.limits.max_turns):
                token.raise_if_cancelled()
                budget_failure = self._budget_failure(spec, usage)
                if budget_failure is not None:
                    return self._failed(
                        workspace,
                        usage,
                        audit,
                        stop_reason=budget_failure[0],
                        failure=budget_failure[1],
                    )

                turn = workspace.advance_turn()
                audit.append("turn_started", {"turn": turn})
                steering = self._drain_steering(controls)
                for item in steering:
                    audit.append(
                        "steering_applied",
                        {
                            "turn": turn,
                            "input_id": item.input_id,
                            "content": item.content,
                        },
                    )
                context = await self._build_context(
                    workspace,
                    cancellation=token,
                    turn=turn,
                    transient_instructions=tuple(
                        TransientInstruction(item.content, "steering")
                        for item in steering
                    ),
                )
                audit.append(
                    "context_built",
                    {
                        "turn": turn,
                        "source_revision": context.source_revision,
                        "message_count": len(context.messages),
                        "metadata": context.metadata,
                    },
                )
                usage.begin_request()
                completed = await self._request_model(
                    ModelRequest(
                        messages=context.messages,
                        tools=definitions,
                    ),
                    cancellation=token,
                    audit=audit,
                    turn=turn,
                )
                usage.record(completed.usage)
                message = completed.message
                try:
                    workspace.append_assistant(message)
                except ValueError as exc:
                    raise ModelProtocolError(str(exc)) from exc
                audit.append(
                    "assistant_message",
                    self._assistant_payload(message, turn),
                )

                if not message.tool_calls:
                    assert message.content is not None
                    audit.append("turn_completed", {"turn": turn})
                    return self._terminal(
                        workspace,
                        usage,
                        audit,
                        status=RunStatus.COMPLETED,
                        stop_reason=StopReason.TEXT_RESPONSE,
                        output=message.content,
                    )

                for call in message.tool_calls:
                    if workspace.record_tool_call(call):
                        audit.append("turn_completed", {"turn": turn})
                        return self._failed(
                            workspace,
                            usage,
                            audit,
                            stop_reason=StopReason.REPEATED_TOOL_CALL,
                            failure=RunFailure(
                                phase=RunPhase.TOOL,
                                code=FailureCode.TOOL_ERROR,
                                message=(
                                    f"tool {call.name!r} was called with the same "
                                    "arguments too many consecutive times"
                                ),
                            ),
                        )
                    execution = await self._execute_tool(
                        call,
                        known=call.name in tool_names,
                        cancellation=token,
                        audit=audit,
                        turn=turn,
                    )
                    workspace.append_tool_result(
                        call,
                        result=execution.result,
                        is_error=execution.is_error,
                    )
                    if execution.control is not ToolControl.CONTINUE:
                        audit.append("turn_completed", {"turn": turn})
                        return self._tool_terminal(
                            workspace,
                            usage,
                            audit,
                            execution,
                        )
                audit.append("turn_completed", {"turn": turn})

            return self._failed(
                workspace,
                usage,
                audit,
                stop_reason=StopReason.MAX_STEPS,
                failure=RunFailure(
                    phase=RunPhase.CONTROL,
                    code=FailureCode.BUDGET_EXCEEDED,
                    message=(
                        f"Run {spec.run_id!r} did not finish within "
                        f"{spec.limits.max_turns} turns"
                    ),
                ),
            )
        except RunCancelledError as exc:
            return self._terminal(
                workspace,
                usage,
                audit,
                status=RunStatus.CANCELLED,
                stop_reason=StopReason.EXTERNAL_ABORT,
                output=str(exc),
            )
        except ContextBuildError as exc:
            return self._failed(
                workspace,
                usage,
                audit,
                stop_reason=self._context_stop_reason(exc.code),
                failure=RunFailure(
                    phase=RunPhase.CONTEXT,
                    code=exc.code,
                    message=str(exc),
                    retryable=exc.retryable,
                    cause=exc,
                ),
            )
        except ModelCallError as exc:
            return self._failed(
                workspace,
                usage,
                audit,
                stop_reason=self._model_stop_reason(exc.code),
                failure=RunFailure(
                    phase=RunPhase.MODEL,
                    code=exc.code,
                    message=str(exc),
                    retryable=exc.retryable,
                    cause=exc,
                ),
            )
        except ToolExecutionError as exc:
            return self._failed(
                workspace,
                usage,
                audit,
                stop_reason=StopReason.RUNTIME_ERROR,
                failure=RunFailure(
                    phase=RunPhase.TOOL,
                    code=FailureCode.TOOL_ERROR,
                    message=str(exc),
                    retryable=exc.retryable,
                    cause=exc,
                ),
            )

    async def _build_context(
        self,
        workspace: _RunWorkspace,
        *,
        cancellation: CancellationToken,
        turn: int,
        transient_instructions: tuple[TransientInstruction, ...],
    ) -> ContextView:
        request = ContextRequest(
            run_id=workspace.spec.run_id,
            source_revision=workspace.spec.base_revision,
            turn=turn,
            committed_messages=workspace.committed_messages,
            pending_messages=workspace.pending_messages,
            transient_instructions=transient_instructions,
            metadata=workspace.spec.metadata,
        )
        try:
            view = await cancellation.run(
                self._context.build(request, cancellation=cancellation)
            )
        except (RunCancelledError, ContextBuildError, ContextProtocolError):
            raise
        except Exception as exc:
            raise ContextProtocolError(
                f"ContextPipeline raised an undeclared {type(exc).__name__}"
            ) from exc
        if not isinstance(view, ContextView):
            raise ContextProtocolError(
                "ContextPipeline.build() must return ContextView"
            )
        if (
            view.run_id != request.run_id
            or view.source_revision != request.source_revision
            or view.turn != request.turn
        ):
            raise ContextProtocolError(
                "ContextView identity does not match its ContextRequest"
            )
        return view

    @staticmethod
    def _drain_steering(
        controls: RunControlSource | None,
    ) -> tuple[SteeringInput, ...]:
        if controls is None:
            return ()
        items = controls.drain_steering()
        if not isinstance(items, tuple) or not all(
            isinstance(item, SteeringInput) for item in items
        ):
            raise ControlProtocolError(
                "RunControlSource.drain_steering() must return "
                "tuple[SteeringInput, ...]"
            )
        return items

    def _snapshot_tool_definitions(self) -> tuple[ToolDefinition, ...]:
        definitions = tuple(self._tools.definitions)
        if not all(isinstance(item, ToolDefinition) for item in definitions):
            raise ToolProtocolError(
                "ToolExecutor definitions must contain ToolDefinition values"
            )
        names = [definition.name for definition in definitions]
        if len(names) != len(set(names)):
            raise ToolProtocolError("ToolExecutor contains duplicate Tool names")
        return definitions

    async def _request_model(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
        audit: _AuditTrail,
        turn: int,
    ) -> ModelResponseCompleted:
        stream = self._model.stream(request, cancellation=cancellation)
        iterator = stream.__aiter__()
        try:
            while True:
                cancellation.raise_if_cancelled()
                try:
                    event = await cancellation.run(anext(iterator))
                except StopAsyncIteration:
                    break
                if isinstance(event, ModelTextDelta):
                    audit.append(
                        "model_text_delta",
                        {"turn": turn, "delta": event.delta},
                    )
                elif isinstance(event, ModelThinkingDelta):
                    audit.append(
                        "model_thinking_delta",
                        {"turn": turn, "delta": event.delta},
                    )
                elif isinstance(event, ModelResponseCompleted):
                    return event
                else:
                    raise ModelProtocolError(
                        "ModelPort returned an unsupported stream event: "
                        f"{type(event).__name__}"
                    )
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()
        raise ModelProtocolError("ModelPort stream ended without completion")

    async def _execute_tool(
        self,
        call: ToolCall,
        *,
        known: bool,
        cancellation: CancellationToken,
        audit: _AuditTrail,
        turn: int,
    ) -> ToolExecutionResult:
        audit.append(
            "tool_started",
            {
                "turn": turn,
                "tool_call_id": call.id,
                "tool_name": call.name,
                "arguments": call.arguments,
            },
        )
        if not known:
            execution = ToolExecutionResult(
                {
                    "status": "error",
                    "tool": call.name,
                    "error": f"unknown tool {call.name!r}",
                },
                error=f"unknown tool {call.name!r}",
            )
        else:
            try:
                execution = await cancellation.run(
                    self._tools.execute(call, cancellation=cancellation)
                )
            except (RunCancelledError, ToolExecutionError):
                raise
            except Exception as exc:
                raise ToolProtocolError(
                    f"ToolExecutor raised an undeclared {type(exc).__name__}"
                ) from exc
            if not isinstance(execution, ToolExecutionResult):
                raise ToolProtocolError(
                    "ToolExecutor.execute() must return ToolExecutionResult"
                )
        audit.append(
            "tool_completed",
            {
                "turn": turn,
                "tool_call_id": call.id,
                "tool_name": call.name,
                "control": execution.control.value,
                "is_error": execution.is_error,
                "result": execution.result,
            },
        )
        return execution

    def _tool_terminal(
        self,
        workspace: _RunWorkspace,
        usage: _UsageAccumulator,
        audit: _AuditTrail,
        execution: ToolExecutionResult,
    ) -> RunOutcome:
        if execution.control is ToolControl.COMPLETE:
            status = RunStatus.COMPLETED
            reason = StopReason.TOOL_COMPLETION
        elif execution.control is ToolControl.REJECT:
            status = RunStatus.REJECTED
            reason = StopReason.TOOL_REJECTED
        else:
            status = RunStatus.CANCELLED
            reason = StopReason.TOOL_CANCELLED
        return self._terminal(
            workspace,
            usage,
            audit,
            status=status,
            stop_reason=reason,
            output=execution.output,
        )

    def _terminal(
        self,
        workspace: _RunWorkspace,
        usage: _UsageAccumulator,
        audit: _AuditTrail,
        *,
        status: RunStatus,
        stop_reason: StopReason,
        output: str | None,
    ) -> RunOutcome:
        result = RunResult(
            run_id=workspace.spec.run_id,
            status=status,
            stop_reason=stop_reason,
            turns=workspace.turn,
            output=output,
            usage=usage.snapshot(),
        )
        audit.append(
            "run_finished",
            {
                "status": status.value,
                "stop_reason": stop_reason.value,
                "turns": workspace.turn,
            },
        )
        return RunOutcome(
            result=result,
            delta=workspace.delta,
            audit_records=tuple(audit.records),
        )

    def _failed(
        self,
        workspace: _RunWorkspace,
        usage: _UsageAccumulator,
        audit: _AuditTrail,
        *,
        stop_reason: StopReason,
        failure: RunFailure,
    ) -> RunOutcome:
        result = RunResult(
            run_id=workspace.spec.run_id,
            status=RunStatus.FAILED,
            stop_reason=stop_reason,
            turns=workspace.turn,
            usage=usage.snapshot(),
        )
        audit.append(
            "run_finished",
            {
                "status": RunStatus.FAILED.value,
                "stop_reason": stop_reason.value,
                "turns": workspace.turn,
                "failure_code": failure.code.value,
            },
        )
        return RunOutcome(
            result=result,
            delta=workspace.delta,
            audit_records=tuple(audit.records),
            failure=failure,
        )

    @staticmethod
    def _budget_failure(
        spec: RunSpec,
        usage: _UsageAccumulator,
    ) -> tuple[StopReason, RunFailure] | None:
        limit = spec.limits.max_tokens
        if limit is None or usage.request_count == 0:
            return None
        snapshot = usage.snapshot()
        if not snapshot.complete:
            return (
                StopReason.USAGE_UNAVAILABLE,
                RunFailure(
                    phase=RunPhase.CONTROL,
                    code=FailureCode.BUDGET_EXCEEDED,
                    message=(
                        "Run token budget cannot continue because model usage "
                        "was not reported"
                    ),
                ),
            )
        if snapshot.total_tokens >= limit:
            return (
                StopReason.TOKEN_BUDGET_EXCEEDED,
                RunFailure(
                    phase=RunPhase.CONTROL,
                    code=FailureCode.BUDGET_EXCEEDED,
                    message=(
                        f"Run used {snapshot.total_tokens} tokens and cannot start "
                        f"another request under the {limit}-token budget"
                    ),
                ),
            )
        return None

    @staticmethod
    def _model_stop_reason(code: FailureCode) -> StopReason:
        if code is FailureCode.CONTEXT_OVERFLOW:
            return StopReason.CONTEXT_OVERFLOW
        if code is FailureCode.COMPACTION_FAILED:
            return StopReason.COMPACTION_FAILED
        return StopReason.RUNTIME_ERROR

    @staticmethod
    def _context_stop_reason(code: FailureCode) -> StopReason:
        if code is FailureCode.CONTEXT_OVERFLOW:
            return StopReason.CONTEXT_OVERFLOW
        return StopReason.COMPACTION_FAILED

    @staticmethod
    def _assistant_payload(message: AssistantMessage, turn: int) -> JsonObject:
        calls: list[JsonValue] = []
        for call in message.tool_calls:
            calls.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
        return freeze_json_object(
            {
                "turn": turn,
                "content": message.content,
                "tool_calls": calls,
            },
            label="assistant audit payload",
        )
