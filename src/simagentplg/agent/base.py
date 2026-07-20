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
    ControlInput,
    ControlReceipt,
    ControlStatus,
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
        self._operation_lock = asyncio.Lock()
        self._active_operation: _ActiveOperation | None = None
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

    async def startup(self) -> None:
        """Start the model adapter, handlers, and middleware resources."""

        async with self._operation_lock:
            await self._startup()

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

        async with self._operation_lock:
            await self._shutdown()

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

        async with self._operation_lock:
            await self._startup()
            return await self._tool_runtime.dispatch(tool_name, arguments)

    async def run(self, *, task: str) -> AgentRunResult:
        """Run one task and return a structured terminal result."""

        self._pending_operations += 1
        self._idle_event.clear()
        try:
            async with self._operation_lock:
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
        finally:
            self._pending_operations -= 1
            if self._pending_operations == 0:
                self._idle_event.set()

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
        self._pending_operations += 1
        self._idle_event.clear()
        try:
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
            self._pending_operations -= 1
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
