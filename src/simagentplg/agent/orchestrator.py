from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any, Protocol

from simagentplg.agent.behavior import (
    BehaviorAction,
    BehaviorDecision,
    BehaviorHook,
    BehaviorHookError,
    TurnSnapshot,
)
from simagentplg.agent.cancellation import (
    AgentCancelledError,
    CancellationSource,
    CancellationToken,
)
from simagentplg.agent.compaction import (
    CompactionRuntime,
    CompactionStatus,
    CompactionTrigger,
    Compactor,
)
from simagentplg.agent.context_builder import (
    AgentContextBuilder,
    ContextBuildResult,
)
from simagentplg.agent.context_management import (
    AutoCompactionPolicy,
    CompactionPolicy,
    CompactionPreparation,
    MessageTokenEstimator,
    estimate_context_usage,
)
from simagentplg.agent.control import (
    ContinueRejectedError,
    _continue_history_rejection_reason,
    _SteeringQueue,
)
from simagentplg.agent.events import (
    AgentContinued,
    AgentEventEmitter,
    AgentFinished,
    AgentStarted,
    AssistantTextDelta,
    AssistantThinkingDelta,
    ContextPressureEvaluated,
    MessageCompleted,
    SteeringApplied,
    SteeringDiscarded,
    TurnCompleted,
    TurnStarted,
)
from simagentplg.agent.result import AgentRunResult, RunStatus, StopReason
from simagentplg.agent.runtime_policy import RuntimePolicy
from simagentplg.agent.state import AgentState
from simagentplg.agent.tool_runtime import (
    RepeatedToolCallError,
    ToolRuntime,
)
from simagentplg.agent.types import ToolCallResult, ToolControl
from simagentplg.agent.usage import UsageAccumulator
from simagentplg.handlers.definition import ToolDefinition, ToolEffect
from simagentplg.plugins.skill.skill_manager import SkillManager
from simagentplg.providers.base import (
    AssistantMessage,
    ContextOverflowError,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelToolCall,
    serialize_assistant_message,
)

TOOL_COMPLETION_RETRY_PROMPT = """
Explicit-finish mode requires a completing tool call to finish the task.
If the work is complete, call a tool that returns completion control now.
Do not end with plain text.
""".strip()


class ModelStream(Protocol):
    """Provider stream shape consumed by the orchestrator."""

    def __call__(
        self,
        context: ContextBuildResult,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...


class _AutomaticCompactionError(RuntimeError):
    """Automatic compaction failed before provider dispatch or retry."""


class AgentOrchestrator:
    """Coordinate one agent task across model, state, and tool runtimes."""

    def __init__(
        self,
        *,
        agent_id: str,
        state: AgentState,
        context_builder: AgentContextBuilder,
        model_stream: ModelStream,
        tool_runtime: ToolRuntime,
        skill_manager: SkillManager | None,
        policy: RuntimePolicy,
        compaction_policy: CompactionPolicy | None = None,
        auto_compaction_policy: AutoCompactionPolicy | None = None,
        compactor: Compactor | None = None,
        compaction_runtime: CompactionRuntime | None = None,
        context_token_estimator: MessageTokenEstimator | None = None,
        behavior_hooks: tuple[BehaviorHook, ...] = (),
        event_emitter: AgentEventEmitter,
    ) -> None:
        self.agent_id = agent_id
        self.state = state
        self.context_builder = context_builder
        self.model_stream = model_stream
        self.tool_runtime = tool_runtime
        self.skill_manager = skill_manager
        self.policy = policy
        self.compaction_policy = compaction_policy
        self.auto_compaction_policy = auto_compaction_policy
        self.compactor = compactor
        self.compaction_runtime = compaction_runtime
        self.context_token_estimator = context_token_estimator
        self.behavior_hooks = behavior_hooks
        self.event_emitter = event_emitter
        self._usage = UsageAccumulator()

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return every tool definition available to the model."""

        return self.tool_runtime.tools

    @property
    def tool_definitions(self) -> tuple[ToolDefinition, ...]:
        """Return the canonical tools used by Core routing and scheduling."""

        return self.tool_runtime.tool_definitions

    async def run(
        self,
        *,
        task: str,
        cancellation: CancellationToken | None = None,
        steering: _SteeringQueue | None = None,
    ) -> AgentRunResult:
        """Run one task and return its structured terminal result."""

        return await self._run(
            task=task,
            cancellation=cancellation,
            steering=steering,
        )

    async def continue_run(
        self,
        *,
        cancellation: CancellationToken | None = None,
        steering: _SteeringQueue | None = None,
    ) -> AgentRunResult:
        """Resume existing history in a distinct Run without a user message."""

        reason = _continue_history_rejection_reason(
            self.state.messages,
            self.state.last_run_result,
        )
        if reason is not None:
            raise ContinueRejectedError(reason)
        return await self._run(
            task=None,
            cancellation=cancellation,
            steering=steering,
        )

    async def _run(
        self,
        *,
        task: str | None,
        cancellation: CancellationToken | None,
        steering: _SteeringQueue | None,
    ) -> AgentRunResult:
        """Execute one task or Continue Run through the shared lifecycle."""

        token = cancellation or CancellationSource().token
        self._usage = UsageAccumulator()
        run_id = self.event_emitter.begin_run()
        try:
            caller_cancellation: asyncio.CancelledError | None = None
            try:
                if task is None:
                    self.state.begin_continue()
                    await self.event_emitter.emit(AgentContinued())
                else:
                    self.state.begin_task(task)
                    await self.event_emitter.emit(AgentStarted(task))
                await token.run(self._prepare_task())
                result = await self._run_loop(token, steering, run_id=run_id)
            except AgentCancelledError as exc:
                result = self._cancelled(str(exc))
            except RepeatedToolCallError as exc:
                result = self._failure(
                    StopReason.REPEATED_TOOL_CALL,
                    str(exc),
                )
            except ContextOverflowError as exc:
                result = self._failure(StopReason.CONTEXT_OVERFLOW, str(exc))
            except _AutomaticCompactionError as exc:
                result = self._failure(StopReason.COMPACTION_FAILED, str(exc))
            except asyncio.CancelledError as exc:
                caller_cancellation = exc
                result = self._cancelled("agent run coroutine was cancelled")
            except Exception as exc:
                result = self._failure(StopReason.RUNTIME_ERROR, str(exc))
            if steering is not None:
                for control in steering.close():
                    await self.event_emitter.emit(
                        SteeringDiscarded(control, result.stop_reason)
                    )
            self._commit_result(result)
            await self.event_emitter.emit(AgentFinished(result))
            if caller_cancellation is not None:
                raise caller_cancellation
            return result
        finally:
            self.event_emitter.end_run(run_id)

    async def _prepare_task(self) -> None:
        await self.tool_runtime.on_task_start()
        self._activate_explicit_skill()

    async def _run_loop(
        self,
        cancellation: CancellationToken,
        steering: _SteeringQueue | None = None,
        *,
        run_id: str,
    ) -> AgentRunResult:
        for _ in range(self.policy.max_steps):
            cancellation.raise_if_cancelled()
            budget_failure = self._budget_failure()
            if budget_failure is not None:
                return budget_failure
            turn = self.state.advance_turn()
            await self.event_emitter.emit(TurnStarted(turn))
            try:
                response = await self._chat_next_turn(cancellation, steering)
                message = response.message
                self._usage.record(response.usage)
                self.state.add_message(
                    serialize_assistant_message(
                        message,
                        usage=response.usage,
                    )
                )
                await self.event_emitter.emit(
                    MessageCompleted(turn, message, response.usage)
                )
                cancellation.raise_if_cancelled()

                if not message.tool_calls:
                    if steering is not None and steering.has_pending:
                        self.state.no_tool_response_count = 0
                    elif not self.policy.require_explicit_finish:
                        if message.content:
                            return AgentRunResult(
                                status=RunStatus.COMPLETED,
                                stop_reason=StopReason.TEXT_RESPONSE,
                                turns=self.state.turn,
                                output=message.content,
                                usage=self._usage.snapshot(),
                            )
                        return self._failure(
                            StopReason.EMPTY_RESPONSE,
                            "chat completion returned empty content",
                        )

                    else:
                        self.state.no_tool_response_count += 1
                        if (
                            self.state.no_tool_response_count
                            >= self.policy.max_no_tool_responses
                        ):
                            return self._failure(
                                StopReason.MAX_NO_TOOL_RESPONSES,
                                "explicit-finish mode produced plain text without a "
                                "completing tool call "
                                f"{self.state.no_tool_response_count} "
                                "consecutive times",
                            )
                else:
                    self.state.no_tool_response_count = 0
                    tool_result = await self._execute_tool_calls(
                        message,
                        cancellation,
                    )
                    self.state.add_messages(list(tool_result.messages))
                    if tool_result.cancelled:
                        raise AgentCancelledError(
                            tool_result.error
                            or cancellation.reason
                            or "agent run was aborted"
                        )
                    terminal_result = self._terminal_tool_result(tool_result)
                    if terminal_result is not None:
                        return terminal_result
            finally:
                await self.event_emitter.emit(TurnCompleted(turn))

            behavior_result = await self._evaluate_after_turn(
                run_id=run_id,
                turn=turn,
                response=message,
                cancellation=cancellation,
            )
            if behavior_result is not None:
                return behavior_result

        return self._failure(
            StopReason.MAX_STEPS,
            f"agent {self.agent_id!r} did not finish within "
            f"{self.policy.max_steps} steps",
        )

    async def _evaluate_after_turn(
        self,
        *,
        run_id: str,
        turn: int,
        response: AssistantMessage,
        cancellation: CancellationToken,
    ) -> AgentRunResult | None:
        for hook in self.behavior_hooks:
            snapshot = TurnSnapshot(
                agent_id=self.agent_id,
                run_id=run_id,
                turn=turn,
                task=self.state.task,
                response=response,
                usage=self._usage.snapshot(),
                messages=self.state.messages,
            )
            try:
                decision = await cancellation.run(
                    hook.after_turn(snapshot, cancellation=cancellation)
                )
                if decision is not None and not isinstance(decision, BehaviorDecision):
                    raise TypeError("after_turn must return BehaviorDecision or None")
            except (AgentCancelledError, asyncio.CancelledError):
                raise
            except Exception as exc:
                raise BehaviorHookError(hook, exc) from exc

            if decision is not None and decision.action is BehaviorAction.STOP:
                return AgentRunResult(
                    status=RunStatus.COMPLETED,
                    stop_reason=StopReason.BEHAVIOR_STOP,
                    turns=self.state.turn,
                    output=(
                        decision.output
                        if decision.output is not None
                        else response.content
                    ),
                    usage=self._usage.snapshot(),
                )
        return None

    async def _chat_next_turn(
        self,
        cancellation: CancellationToken,
        steering: _SteeringQueue | None = None,
    ) -> ModelResponseCompleted:
        context, preparation = await self._build_evaluated_context(steering)
        auto_policy = self.auto_compaction_policy
        if (
            auto_policy is not None
            and auto_policy.enabled
            and auto_policy.compact_on_pressure
            and preparation is not None
        ):
            result = await self._compact_automatically(
                cancellation,
                trigger=CompactionTrigger.PRESSURE,
                preparation=preparation,
            )
            if result:
                context, _ = await self._build_evaluated_context(steering)

        overflow_retries = 0
        while True:
            try:
                return await self._request_model(context, cancellation)
            except ContextOverflowError as exc:
                if (
                    exc.response_started
                    or auto_policy is None
                    or not auto_policy.enabled
                    or not auto_policy.recover_on_overflow
                    or overflow_retries >= auto_policy.max_overflow_retries
                ):
                    raise
                compacted = await self._compact_automatically(
                    cancellation,
                    trigger=CompactionTrigger.OVERFLOW,
                )
                if not compacted:
                    raise
                overflow_retries += 1
                context, _ = await self._build_evaluated_context(steering)

    async def _build_evaluated_context(
        self,
        steering: _SteeringQueue | None = None,
    ) -> tuple[ContextBuildResult, CompactionPreparation | None]:
        while True:
            await self._apply_steering(steering)
            context = self.context_builder.build(
                self.state,
                tools=self.tool_definitions,
                transient_messages=self._runtime_context_messages(),
            )
            preparation = await self._evaluate_context_pressure(context)
            if steering is None or not steering.has_pending:
                return context, preparation

    async def _apply_steering(self, steering: _SteeringQueue | None) -> None:
        if steering is None:
            return
        controls = steering.drain()
        if not controls:
            return
        self.state.no_tool_response_count = 0
        for control in controls:
            self.state.add_message({"role": "user", "content": control.content})
            await self.event_emitter.emit(
                SteeringApplied(control=control, target_turn=self.state.turn)
            )
        self._activate_explicit_skill()

    async def _request_model(
        self,
        context: ContextBuildResult,
        cancellation: CancellationToken,
    ) -> ModelResponseCompleted:
        self._usage.begin_request()
        stream = self.model_stream(
            context,
            cancellation=cancellation,
        )
        iterator = stream.__aiter__()
        completed: ModelResponseCompleted | None = None
        response_started = False
        try:
            while completed is None:
                cancellation.raise_if_cancelled()
                try:
                    event = await cancellation.run(anext(iterator))
                except StopAsyncIteration:
                    break

                if isinstance(event, ModelTextDelta):
                    response_started = True
                    await self.event_emitter.emit(
                        AssistantTextDelta(self.state.turn, event.delta)
                    )
                elif isinstance(event, ModelThinkingDelta):
                    response_started = True
                    await self.event_emitter.emit(
                        AssistantThinkingDelta(
                            self.state.turn,
                            event.delta,
                        )
                    )
                elif isinstance(event, ModelResponseCompleted):
                    completed = event
                else:
                    raise TypeError(
                        "model stream returned an unsupported event: "
                        f"{type(event).__name__}"
                    )
        except ContextOverflowError as exc:
            if response_started and not exc.response_started:
                raise ContextOverflowError(
                    str(exc),
                    response_started=True,
                ) from exc
            raise
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()

        if completed is None:
            raise RuntimeError("model stream ended without a completed response")
        return completed

    async def _evaluate_context_pressure(
        self,
        context: ContextBuildResult,
    ) -> CompactionPreparation | None:
        policy = self.compaction_policy
        if policy is None:
            return None

        estimate = estimate_context_usage(
            context.agent_messages,
            tools=context.tools,
            estimator=self.context_token_estimator,
        )
        decision = policy.evaluate(estimate)
        preparation = (
            policy.prepare(
                self.state.messages,
                estimator=self.context_token_estimator,
            )
            if decision.should_compact
            else None
        )
        await self.event_emitter.emit(
            ContextPressureEvaluated(
                turn=self.state.turn,
                decision=decision,
                preparation=preparation,
            )
        )
        return preparation

    async def _compact_automatically(
        self,
        cancellation: CancellationToken,
        *,
        trigger: CompactionTrigger,
        preparation: CompactionPreparation | None = None,
    ) -> bool:
        runtime = self.compaction_runtime
        compactor = self.compactor
        if runtime is None or compactor is None:
            raise _AutomaticCompactionError(
                "automatic compaction is missing its runtime or Compactor"
            )
        result = await runtime.compact_in_active_run(
            compactor,
            cancellation=cancellation,
            trigger=trigger,
            preparation=preparation,
        )
        if result.status is CompactionStatus.CANCELLED:
            raise AgentCancelledError(
                result.error or cancellation.reason or "agent run was aborted"
            )
        if result.status is CompactionStatus.FAILED:
            raise _AutomaticCompactionError(
                result.error or "automatic compaction failed"
            )
        return result.status is CompactionStatus.COMPLETED

    def _budget_failure(self) -> AgentRunResult | None:
        limit = self.policy.max_run_tokens
        if limit is None:
            return None

        usage = self._usage.snapshot()
        if usage.request_count == 0:
            return None
        if not usage.complete:
            return self._failure(
                StopReason.USAGE_UNAVAILABLE,
                "run token budget cannot continue because "
                f"{usage.missing_request_count} model request(s) did not "
                "report usage",
            )
        if usage.total_tokens >= limit:
            return self._failure(
                StopReason.TOKEN_BUDGET_EXCEEDED,
                f"run used {usage.total_tokens} tokens and cannot start "
                f"another model request under the {limit}-token budget",
            )
        return None

    def _runtime_context_messages(self) -> list[dict[str, str]]:
        if self.state.no_tool_response_count == 0:
            return []
        return [
            {
                "role": "system",
                "content": self._tool_completion_retry_prompt(
                    self.state.no_tool_response_count
                ),
            }
        ]

    def _tool_completion_retry_prompt(
        self,
        no_tool_response_count: int,
    ) -> str:
        if no_tool_response_count <= 1:
            return TOOL_COMPLETION_RETRY_PROMPT
        return (
            TOOL_COMPLETION_RETRY_PROMPT
            + "\n\n"
            + (
                f"Retry {no_tool_response_count}/"
                f"{self.policy.max_no_tool_responses}: "
                "the previous response still did not include a tool call."
            )
        )

    def _activate_explicit_skill(self) -> None:
        if self.skill_manager is None:
            return

        skill_name = self.skill_manager.select_explicit_skill(self.state.messages)
        if skill_name is not None:
            self.state.active_skill_name = skill_name

    async def _execute_tool_calls(
        self,
        message: AssistantMessage,
        cancellation: CancellationToken,
    ) -> ToolCallResult:
        result_messages: list[dict[str, Any]] = []

        tool_calls = message.tool_calls or ()
        index = 0
        while index < len(tool_calls):
            tool_call = tool_calls[index]
            next_index = index + 1
            if (
                self.policy.parallel_tool_calls
                and self.tool_runtime.tool_effect(tool_call.name)
                is ToolEffect.READ_ONLY
            ):
                while (
                    next_index < len(tool_calls)
                    and self.tool_runtime.tool_effect(tool_calls[next_index].name)
                    is ToolEffect.READ_ONLY
                ):
                    next_index += 1

            batch = tool_calls[index:next_index]
            if len(batch) > 1:
                tool_result = await self._execute_read_only_tool_calls(
                    batch,
                    cancellation,
                )
            else:
                tool_result = await self.tool_runtime.execute_tool_call(
                    tool_call,
                    cancellation=cancellation,
                )
            result_messages.extend(tool_result.messages)
            if tool_result.cancelled:
                reason = (
                    tool_result.error or cancellation.reason or "agent run was aborted"
                )
                for pending_call in tool_calls[next_index:]:
                    pending_result = await self.tool_runtime.cancel_tool_call(
                        pending_call,
                        reason=reason,
                    )
                    result_messages.extend(pending_result.messages)
                return ToolCallResult(
                    tuple(result_messages),
                    error=reason,
                    cancelled=True,
                )
            if tool_result.control is not ToolControl.CONTINUE:
                return ToolCallResult(
                    tuple(result_messages),
                    control=tool_result.control,
                    output=tool_result.output,
                )
            index = next_index

        return ToolCallResult(tuple(result_messages))

    async def _execute_read_only_tool_calls(
        self,
        tool_calls: tuple[ModelToolCall, ...],
        cancellation: CancellationToken,
    ) -> ToolCallResult:
        """Run safe calls concurrently while committing results in source order."""

        result_messages: list[dict[str, Any]] = []
        if not self.tool_runtime.can_parallelize_tool_calls(tool_calls):
            return await self._execute_sequential_tool_calls(
                tool_calls,
                cancellation,
            )
        limit = self.policy.max_parallel_tool_calls or len(tool_calls)
        offset = 0
        while offset < len(tool_calls):
            batch = tool_calls[offset : offset + limit]
            prepared = await self.tool_runtime.prepare_parallel_tool_calls(batch)
            if not prepared:
                return await self._execute_sequential_tool_calls(
                    tool_calls[offset:],
                    cancellation,
                    prefix=result_messages,
                )

            results = await asyncio.gather(
                *(
                    self.tool_runtime.execute_tool_call(
                        tool_call,
                        cancellation=cancellation,
                        prepared=True,
                        defer_completion=True,
                    )
                    for tool_call in batch
                )
            )
            for tool_call, result in zip(batch, results, strict=True):
                await self.tool_runtime.complete_tool_call(tool_call, result)
                result_messages.extend(result.messages)

            cancelled = next((result for result in results if result.cancelled), None)
            if cancelled is not None:
                reason = (
                    cancelled.error or cancellation.reason or "agent run was aborted"
                )
                for pending_call in tool_calls[offset + len(batch) :]:
                    pending_result = await self.tool_runtime.cancel_tool_call(
                        pending_call,
                        reason=reason,
                    )
                    result_messages.extend(pending_result.messages)
                return ToolCallResult(
                    tuple(result_messages),
                    error=reason,
                    cancelled=True,
                )

            terminal = next(
                (
                    result
                    for result in results
                    if result.control is not ToolControl.CONTINUE
                ),
                None,
            )
            if terminal is not None:
                reason = "skipped after a terminal read-only tool result"
                for pending_call in tool_calls[offset + len(batch) :]:
                    pending_result = await self.tool_runtime.cancel_tool_call(
                        pending_call,
                        reason=reason,
                    )
                    result_messages.extend(pending_result.messages)
                return ToolCallResult(
                    tuple(result_messages),
                    control=terminal.control,
                    output=terminal.output,
                )
            offset += len(batch)

        return ToolCallResult(tuple(result_messages))

    async def _execute_sequential_tool_calls(
        self,
        tool_calls: tuple[ModelToolCall, ...],
        cancellation: CancellationToken,
        *,
        prefix: list[dict[str, Any]] | None = None,
    ) -> ToolCallResult:
        result_messages = list(prefix or ())
        for index, tool_call in enumerate(tool_calls):
            result = await self.tool_runtime.execute_tool_call(
                tool_call,
                cancellation=cancellation,
            )
            result_messages.extend(result.messages)
            if result.cancelled:
                reason = result.error or cancellation.reason or "agent run was aborted"
                for pending_call in tool_calls[index + 1 :]:
                    pending_result = await self.tool_runtime.cancel_tool_call(
                        pending_call,
                        reason=reason,
                    )
                    result_messages.extend(pending_result.messages)
                return ToolCallResult(
                    tuple(result_messages),
                    error=reason,
                    cancelled=True,
                )
            if result.control is not ToolControl.CONTINUE:
                return ToolCallResult(
                    tuple(result_messages),
                    control=result.control,
                    output=result.output,
                )
        return ToolCallResult(tuple(result_messages))

    def _cancelled(self, error: str) -> AgentRunResult:
        return AgentRunResult(
            status=RunStatus.CANCELLED,
            stop_reason=StopReason.EXTERNAL_ABORT,
            turns=self.state.turn,
            error=error,
            usage=self._usage.snapshot(),
        )

    def _terminal_tool_result(
        self,
        tool_result: ToolCallResult,
    ) -> AgentRunResult | None:
        if tool_result.control is ToolControl.CONTINUE:
            return None
        if tool_result.control is ToolControl.COMPLETE:
            return AgentRunResult(
                status=RunStatus.COMPLETED,
                stop_reason=StopReason.TOOL_COMPLETION,
                turns=self.state.turn,
                output=tool_result.output,
                usage=self._usage.snapshot(),
            )
        if tool_result.control is ToolControl.REJECT:
            return AgentRunResult(
                status=RunStatus.REJECTED,
                stop_reason=StopReason.TOOL_REJECTED,
                turns=self.state.turn,
                output=tool_result.output,
                error="tool execution was rejected",
                usage=self._usage.snapshot(),
            )
        return AgentRunResult(
            status=RunStatus.CANCELLED,
            stop_reason=StopReason.TOOL_CANCELLED,
            turns=self.state.turn,
            output=tool_result.output,
            error="tool execution was cancelled",
            usage=self._usage.snapshot(),
        )

    def _failure(
        self,
        stop_reason: StopReason,
        error: str,
    ) -> AgentRunResult:
        return AgentRunResult(
            status=RunStatus.FAILED,
            stop_reason=stop_reason,
            turns=self.state.turn,
            error=error,
            usage=self._usage.snapshot(),
        )

    def _commit_result(self, result: AgentRunResult) -> None:
        self.state.last_run_result = result
        if result.status is RunStatus.COMPLETED:
            self.state.complete(result.output or "")
        elif result.status is RunStatus.REJECTED:
            self.state.reject(result.output)
        elif result.status is RunStatus.CANCELLED:
            self.state.cancel(result.output)
        else:
            self.state.fail(result.error or result.stop_reason.value)
