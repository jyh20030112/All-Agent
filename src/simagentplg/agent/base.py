from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simagentplg.agent.cancellation import (
    CancellationSource,
)
from simagentplg.agent.compaction import (
    CompactionResult,
    CompactionRuntime,
    Compactor,
)
from simagentplg.agent.context_builder import AgentContextBuilder
from simagentplg.agent.context_management import (
    AutoCompactionPolicy,
    CompactionPolicy,
    MessageTokenEstimator,
)
from simagentplg.agent.control import (
    ContinueRejectedError,
    ContinueRejectedReason,
    ControlInput,
    ControlReceipt,
    ControlStatus,
    FollowUpDiscardReason,
    FollowUpFailurePolicy,
    FollowUpHandle,
    _continue_history_rejection_reason,
    _FollowUpQueue,
    _SteeringQueue,
)
from simagentplg.agent.events import (
    AgentEventEmitter,
    AgentEventSink,
)
from simagentplg.agent.orchestrator import AgentOrchestrator
from simagentplg.agent.result import AgentRunResult
from simagentplg.agent.runtime_policy import RuntimePolicy
from simagentplg.agent.state import AgentState
from simagentplg.agent.tool_runtime import ToolRuntime
from simagentplg.agent.types import StepOutcome
from simagentplg.logger import get_logger
from simagentplg.middleware import ToolMiddleware
from simagentplg.plugins.skill.skill_manager import SkillManager
from simagentplg.providers.base import ModelAdapter

if TYPE_CHECKING:
    from simagentplg.handlers.base import BaseHandler
    from simagentplg.session.types import AgentSession

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."

TOOL_PROTOCOL_PROMPT = """
You can call external tools when they are available.

Tool protocol:
- Use tool calls for actions that require a registered tool.
- Wait for tool results before deciding the next action.
- Do not repeat the same ineffective tool call.
""".strip()

EXPLICIT_FINISH_PROTOCOL_PROMPT = """
This agent requires explicit tool completion.
Plain text does not finish the task. After completing all work, call a tool
that returns the completion control signal.
""".strip()


@dataclass(slots=True)
class _ActiveOperation:
    cancellation: CancellationSource
    steering: _SteeringQueue | None = None


class BaseAgent:
    """Stateful agent core composed with a model adapter and tool handlers."""

    def __init__(
        self,
        model: ModelAdapter,
        *,
        agent_id: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        handlers: Iterable[BaseHandler] | None = None,
        middlewares: Iterable[ToolMiddleware] | None = None,
        skills_dir: str | Path | None = None,
        context_builder: AgentContextBuilder | None = None,
        compaction_policy: CompactionPolicy | None = None,
        compactor: Compactor | None = None,
        auto_compaction_policy: AutoCompactionPolicy | None = None,
        context_token_estimator: MessageTokenEstimator | None = None,
        runtime_policy: RuntimePolicy | None = None,
        event_sink: AgentEventSink | None = None,
        steering_queue_capacity: int = 16,
        follow_up_queue_capacity: int = 16,
        follow_up_failure_policy: FollowUpFailurePolicy = (
            FollowUpFailurePolicy.DISCARD
        ),
    ) -> None:
        self._agent_id = agent_id.strip()
        if not self._agent_id:
            raise ValueError("agent_id must not be empty")
        if isinstance(steering_queue_capacity, bool) or not isinstance(
            steering_queue_capacity, int
        ):
            raise TypeError("steering_queue_capacity must be an integer")
        if steering_queue_capacity <= 0:
            raise ValueError("steering_queue_capacity must be greater than zero")
        if isinstance(follow_up_queue_capacity, bool) or not isinstance(
            follow_up_queue_capacity, int
        ):
            raise TypeError("follow_up_queue_capacity must be an integer")
        if follow_up_queue_capacity <= 0:
            raise ValueError("follow_up_queue_capacity must be greater than zero")
        if not isinstance(follow_up_failure_policy, FollowUpFailurePolicy):
            raise TypeError("follow_up_failure_policy must be a FollowUpFailurePolicy")
        policy = runtime_policy or RuntimePolicy()
        self.model = model
        self.system_prompt = system_prompt
        self.runtime_policy = policy
        self.compaction_policy = compaction_policy
        self.compactor = compactor
        self.auto_compaction_policy = auto_compaction_policy
        if auto_compaction_policy is not None and auto_compaction_policy.enabled:
            if compaction_policy is None:
                raise ValueError("automatic compaction requires a CompactionPolicy")
            if compactor is None:
                raise ValueError("automatic compaction requires a Compactor")
        self.context_token_estimator = context_token_estimator
        self.handlers = list(handlers or ())
        self.middlewares = list(middlewares or ())
        self.event_sink = event_sink
        self.steering_queue_capacity = steering_queue_capacity
        self.follow_up_queue_capacity = follow_up_queue_capacity
        self.follow_up_failure_policy = follow_up_failure_policy
        self._operation_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._run_chain_claim_lock = asyncio.Lock()
        self._run_chain_gate = asyncio.Event()
        self._run_chain_gate.set()
        self._active_operation: _ActiveOperation | None = None
        self._follow_ups = _FollowUpQueue(follow_up_queue_capacity)
        self._follow_up_chain_open = False
        self._follow_up_worker: asyncio.Task[None] | None = None
        self._shutting_down = False
        self._pending_operations = 0
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._started = False
        self._skill_manager = SkillManager(skills_dir) if skills_dir else None
        self.state = AgentState()
        self._context_builder = context_builder or AgentContextBuilder(
            skill_manager=self._skill_manager,
        )
        self.logger = get_logger(f"{self.agent_id}")
        self._event_emitter = AgentEventEmitter(
            agent_id=self.agent_id,
            sink=self.event_sink,
            logger=self.logger,
        )
        self._compaction_runtime = CompactionRuntime(
            state=self.state,
            policy=self.compaction_policy,
            estimator=self.context_token_estimator,
            event_emitter=self._event_emitter,
        )
        self._tool_runtime = ToolRuntime(
            self.handlers,
            self.middlewares,
            state=self.state,
            logger=self.logger,
            event_emitter=self._event_emitter,
            max_repeated_tool_calls=policy.max_repeated_tool_calls,
        )
        self.orchestrator = AgentOrchestrator(
            agent_id=self.agent_id,
            state=self.state,
            context_builder=self._context_builder,
            model_stream=self.model.stream,
            tool_runtime=self._tool_runtime,
            skill_manager=self._skill_manager,
            policy=self.runtime_policy,
            compaction_policy=self.compaction_policy,
            auto_compaction_policy=self.auto_compaction_policy,
            compactor=self.compactor,
            compaction_runtime=self._compaction_runtime,
            context_token_estimator=self.context_token_estimator,
            event_emitter=self._event_emitter,
        )
        self.reset()

    @property
    def agent_id(self) -> str:
        """Return the immutable identity of this agent."""

        return self._agent_id

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return the currently registered function tool definitions."""

        return self.orchestrator.tools

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the agent's persistent conversation history."""

        return self.state.messages

    @property
    def pending_steering_count(self) -> int:
        """Return queued Steering inputs for the active Run."""

        active_operation = self._active_operation
        if active_operation is None or active_operation.steering is None:
            return 0
        return active_operation.steering.size

    @property
    def pending_follow_up_count(self) -> int:
        """Return accepted Follow-ups that have not started their Run."""

        return self._follow_ups.size

    @property
    def continue_rejection_reason(self) -> ContinueRejectedReason | None:
        """Return why Continue is currently unavailable, if any."""

        if self._shutting_down:
            return ContinueRejectedReason.AGENT_SHUTTING_DOWN
        if self._pending_operations or not self._run_chain_gate.is_set():
            return ContinueRejectedReason.AGENT_ACTIVE
        return _continue_history_rejection_reason(
            self.state.messages,
            self.state.last_run_result,
        )

    @property
    def can_continue(self) -> bool:
        """Return whether a Continue Run could be started now."""

        return self.continue_rejection_reason is None

    def reset(
        self,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Reset conversation memory while preserving the agent identity."""

        messages = [{"role": "system", "content": self.system_prompt}]
        if self.handlers:
            messages.append({"role": "system", "content": TOOL_PROTOCOL_PROMPT})
        if self.runtime_policy.require_explicit_finish:
            messages.append(
                {
                    "role": "system",
                    "content": EXPLICIT_FINISH_PROTOCOL_PROMPT,
                }
            )
        if history:
            messages.extend(dict(message) for message in history)
        self.state.reset(messages)

    def restore_session(self, session: AgentSession) -> None:
        """Restore one finished Session projection into this Agent's history."""

        if self._pending_operations or self._operation_lock.locked():
            raise RuntimeError("cannot restore a Session while the agent is active")
        snapshot = session.snapshot()
        if snapshot.agent_id is not None and snapshot.agent_id != self.agent_id:
            raise ValueError(
                f"session {snapshot.session_id!r} belongs to agent "
                f"{snapshot.agent_id!r}, not {self.agent_id!r}"
            )
        unfinished = [run.run_id for run in snapshot.runs if not run.finished]
        if unfinished:
            raise ValueError(
                "cannot restore a Session with unfinished run(s): "
                + ", ".join(unfinished)
            )
        self.reset(snapshot.messages)
        if snapshot.runs:
            self.state.last_run_result = snapshot.runs[-1].result

    async def startup(self) -> None:
        """Start the model adapter, handlers, and middleware resources."""

        await self._claim_run_chain()
        try:
            async with self._operation_lock:
                await self._startup()
        finally:
            self._release_run_chain()

    async def _startup(self) -> None:
        await self._ensure_skills_discovered()

        if self._started:
            return

        try:
            await self.model.startup()
            await self._tool_runtime.startup()
            if self._tool_runtime.tools:
                self.logger.info(
                    "Loaded %d handler(s); registered tools: %s",
                    len(self.handlers),
                    ", ".join(
                        sorted(
                            tool["function"]["name"]
                            for tool in self._tool_runtime.tools
                        )
                    ),
                )
        except Exception:
            try:
                await self._tool_runtime.shutdown()
            except Exception as shutdown_error:
                self.logger.warning(
                    "Tool runtime rollback shutdown failed: %s",
                    shutdown_error,
                )
            try:
                await self.model.shutdown()
            except Exception as shutdown_error:
                self.logger.warning(
                    "Model adapter rollback shutdown failed: %s",
                    shutdown_error,
                )
            raise

        self._started = True

    async def shutdown(self) -> None:
        """Release all resources owned by this agent."""

        async with self._shutdown_lock:
            self._shutting_down = True
            self._discard_follow_ups(FollowUpDiscardReason.AGENT_SHUTDOWN)
            chain_claimed = False
            try:
                await self._claim_run_chain()
                chain_claimed = True
                async with self._operation_lock:
                    await self._shutdown()
            finally:
                self._shutting_down = False
                if chain_claimed:
                    self._release_run_chain()

    async def _shutdown(self) -> None:
        if not self._started:
            return

        errors: list[Exception] = []
        try:
            await self._tool_runtime.shutdown()
        except Exception as exc:
            errors.append(exc)
        try:
            await self.model.shutdown()
        except Exception as exc:
            errors.append(exc)
        self._started = False
        if errors:
            raise RuntimeError(
                f"failed to shut down {len(errors)} agent resource(s)"
            ) from errors[0]

    async def dispatch(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> StepOutcome:
        """Dispatch a tool call to its explicitly registered handler."""

        await self._claim_run_chain()
        try:
            async with self._operation_lock:
                await self._startup()
                return await self._tool_runtime.dispatch(tool_name, arguments)
        finally:
            self._release_run_chain()

    async def run(self, *, task: str) -> AgentRunResult:
        """Run one task and return a structured terminal result."""

        self._track_operation()
        chain_claimed = False
        result: AgentRunResult | None = None
        try:
            await self._claim_run_chain()
            chain_claimed = True
            self._follow_up_chain_open = True
            async with self._operation_lock:
                result = await self._run_task(task)
                return result
        finally:
            self._settle_operation()
            if chain_claimed:
                self._finish_run_chain(result)

    async def continue_run(self) -> AgentRunResult:
        """Resume existing history in a new Run without adding a user message."""

        reason = self.continue_rejection_reason
        if reason is not None:
            raise ContinueRejectedError(reason)
        if not await self._try_claim_run_chain():
            raise ContinueRejectedError(ContinueRejectedReason.AGENT_ACTIVE)

        tracked = False
        result: AgentRunResult | None = None
        try:
            reason = _continue_history_rejection_reason(
                self.state.messages,
                self.state.last_run_result,
            )
            if reason is not None:
                raise ContinueRejectedError(reason)
            self._track_operation()
            tracked = True
            self._follow_up_chain_open = True
            async with self._operation_lock:
                result = await self._continue_task()
                return result
        finally:
            if tracked:
                self._settle_operation()
                self._finish_run_chain(result)
            else:
                self._release_run_chain()

    async def compact(
        self,
        *,
        compactor: Compactor | None = None,
    ) -> CompactionResult:
        """Explicitly summarize old turns and atomically replace history."""

        active_compactor = self.compactor if compactor is None else compactor
        if active_compactor is None:
            raise RuntimeError("explicit compaction requires a Compactor")
        if self.compaction_policy is None:
            raise RuntimeError("explicit compaction requires a CompactionPolicy")
        self._track_operation()
        chain_claimed = False
        try:
            await self._claim_run_chain()
            chain_claimed = True
            async with self._operation_lock:
                source = CancellationSource()
                active_operation = _ActiveOperation(source)
                self._active_operation = active_operation
                try:
                    await self._startup()
                    return await self._compaction_runtime.compact(
                        active_compactor,
                        cancellation=source.token,
                    )
                finally:
                    if self._active_operation is active_operation:
                        self._active_operation = None
        finally:
            self._settle_operation()
            if chain_claimed:
                self._release_run_chain()

    async def _run_task(self, task: str) -> AgentRunResult:
        source = CancellationSource()
        steering = _SteeringQueue(self.steering_queue_capacity)
        active_operation = _ActiveOperation(source, steering)
        self._active_operation = active_operation
        try:
            await self._startup()
            return await self.orchestrator.run(
                task=task,
                cancellation=source.token,
                steering=steering,
            )
        finally:
            steering.close()
            if self._active_operation is active_operation:
                self._active_operation = None

    async def _continue_task(self) -> AgentRunResult:
        source = CancellationSource()
        steering = _SteeringQueue(self.steering_queue_capacity)
        active_operation = _ActiveOperation(source, steering)
        self._active_operation = active_operation
        try:
            await self._startup()
            return await self.orchestrator.continue_run(
                cancellation=source.token,
                steering=steering,
            )
        finally:
            steering.close()
            if self._active_operation is active_operation:
                self._active_operation = None

    async def _claim_run_chain(self) -> None:
        while True:
            await self._run_chain_gate.wait()
            async with self._run_chain_claim_lock:
                if self._run_chain_gate.is_set():
                    self._run_chain_gate.clear()
                    return

    async def _try_claim_run_chain(self) -> bool:
        async with self._run_chain_claim_lock:
            if not self._run_chain_gate.is_set():
                return False
            self._run_chain_gate.clear()
            return True

    def _release_run_chain(self) -> None:
        self._follow_up_chain_open = False
        self._run_chain_gate.set()

    def _finish_run_chain(self, result: AgentRunResult | None) -> None:
        if self._shutting_down:
            self._discard_follow_ups(FollowUpDiscardReason.AGENT_SHUTDOWN)
            self._release_run_chain()
            return
        can_continue = (
            result is not None and result.succeeded
        ) or self.follow_up_failure_policy is FollowUpFailurePolicy.CONTINUE
        if not can_continue:
            self._discard_follow_ups(FollowUpDiscardReason.PREVIOUS_RUN_NOT_COMPLETED)
            self._release_run_chain()
            return
        if self._follow_ups.size == 0:
            self._release_run_chain()
            return
        self._follow_up_worker = asyncio.create_task(
            self._drain_follow_ups(),
            name=f"{self.agent_id}-follow-ups",
        )

    async def _drain_follow_ups(self) -> None:
        try:
            while self._follow_up_chain_open and not self._shutting_down:
                handle = self._follow_ups.pop()
                if handle is None:
                    break
                result: AgentRunResult | None = None
                try:
                    async with self._operation_lock:
                        result = await self._run_task(handle.control.content)
                except asyncio.CancelledError:
                    handle._discard(FollowUpDiscardReason.AGENT_SHUTDOWN)
                    self._discard_follow_ups(FollowUpDiscardReason.AGENT_SHUTDOWN)
                    raise
                except Exception as exc:
                    handle._set_exception(exc)
                else:
                    handle._set_result(result)
                finally:
                    self._settle_operation()

                can_continue = (
                    result is not None and result.succeeded
                ) or self.follow_up_failure_policy is FollowUpFailurePolicy.CONTINUE
                if not can_continue:
                    self._discard_follow_ups(
                        FollowUpDiscardReason.PREVIOUS_RUN_NOT_COMPLETED
                    )
                    break
        finally:
            if self._shutting_down:
                self._discard_follow_ups(FollowUpDiscardReason.AGENT_SHUTDOWN)
            self._follow_up_worker = None
            self._release_run_chain()

    def _discard_follow_ups(self, reason: FollowUpDiscardReason) -> None:
        for handle in self._follow_ups.drain():
            handle._discard(reason)
            self._settle_operation()

    def _track_operation(self) -> None:
        self._pending_operations += 1
        self._idle_event.clear()

    def _settle_operation(self) -> None:
        self._pending_operations -= 1
        if self._pending_operations < 0:
            raise RuntimeError("agent operation accounting became negative")
        if self._pending_operations == 0:
            self._idle_event.set()

    def abort(self, reason: str | None = None) -> bool:
        """Cancel the active run or compaction without waiting for it."""

        active_operation = self._active_operation
        if active_operation is None:
            return False
        return active_operation.cancellation.cancel(reason)

    async def steer(self, content: str) -> ControlReceipt:
        """Queue guidance for the active Run's next model-call safe point."""

        control = ControlInput.steering(content)
        active_operation = self._active_operation
        if active_operation is None or active_operation.steering is None:
            return ControlReceipt(
                control=control,
                status=ControlStatus.AGENT_IDLE,
                queue_size=0,
            )
        if active_operation.cancellation.token.cancelled:
            return ControlReceipt(
                control=control,
                status=ControlStatus.RUN_CLOSING,
                queue_size=active_operation.steering.size,
            )
        return active_operation.steering.submit(control)

    async def follow_up(self, task: str) -> FollowUpHandle:
        """Queue one independent Run after the active Run chain."""

        control = ControlInput.follow_up(task)
        if self._shutting_down:
            return self._follow_ups.rejected(control, ControlStatus.RUN_CLOSING)
        active_operation = self._active_operation
        if (
            active_operation is not None
            and active_operation.cancellation.token.cancelled
        ):
            return self._follow_ups.rejected(control, ControlStatus.RUN_CLOSING)
        if not self._follow_up_chain_open:
            return self._follow_ups.rejected(control, ControlStatus.AGENT_IDLE)
        handle = self._follow_ups.submit(control)
        if handle.accepted:
            self._track_operation()
        return handle

    async def wait_for_idle(self) -> None:
        """Wait for requested runs or compactions and their sinks to settle."""

        await self._idle_event.wait()

    async def runtime(self, *, task: str) -> str | None:
        """Compatibility wrapper returning completed output as text."""

        result = await self.run(task=task)
        result.raise_for_status()
        return result.output

    async def _ensure_skills_discovered(self) -> None:
        if self._skill_manager is not None:
            await self._skill_manager.discover()
