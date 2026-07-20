from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol, TypeAlias
from uuid import uuid4

from simagentplg.agent.compaction import (
    CompactionRequest,
    CompactionResult,
    CompactionStatus,
)
from simagentplg.agent.context_management import (
    CompactionDecision,
    CompactionPreparation,
)
from simagentplg.agent.control import ControlInput, ControlInputKind
from simagentplg.agent.result import AgentRunResult, StopReason
from simagentplg.agent.types import ToolCallResult, ToolProgressUpdate
from simagentplg.providers.base import (
    AssistantMessage,
    ModelToolCall,
    ModelUsage,
)


class AgentEventKind(StrEnum):
    """Stable discriminator for one observable lifecycle event."""

    AGENT_STARTED = "agent_started"
    TURN_STARTED = "turn_started"
    STEERING_APPLIED = "steering_applied"
    STEERING_DISCARDED = "steering_discarded"
    CONTEXT_PRESSURE_EVALUATED = "context_pressure_evaluated"
    COMPACTION_STARTED = "compaction_started"
    COMPACTION_COMPLETED = "compaction_completed"
    COMPACTION_FAILED = "compaction_failed"
    MESSAGE_COMPLETED = "message_completed"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_THINKING_DELTA = "assistant_thinking_delta"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESSED = "tool_progressed"
    TOOL_COMPLETED = "tool_completed"
    TURN_COMPLETED = "turn_completed"
    AGENT_FINISHED = "agent_finished"


@dataclass(frozen=True, slots=True)
class AgentStarted:
    """A task entered the agent runtime."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.AGENT_STARTED
    task: str


@dataclass(frozen=True, slots=True)
class TurnStarted:
    """One provider turn started."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.TURN_STARTED
    turn: int


@dataclass(frozen=True, slots=True)
class SteeringApplied:
    """One queued Steering input entered persistent history at a safe point."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.STEERING_APPLIED
    control: ControlInput
    target_turn: int

    def __post_init__(self) -> None:
        if self.control.kind is not ControlInputKind.STEERING:
            raise ValueError("SteeringApplied requires a Steering control input")
        if self.target_turn <= 0:
            raise ValueError("target_turn must be greater than zero")


@dataclass(frozen=True, slots=True)
class SteeringDiscarded:
    """One accepted Steering input was not applied before its Run ended."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.STEERING_DISCARDED
    control: ControlInput
    reason: StopReason

    def __post_init__(self) -> None:
        if self.control.kind is not ControlInputKind.STEERING:
            raise ValueError("SteeringDiscarded requires a Steering control input")


@dataclass(frozen=True, slots=True)
class ContextPressureEvaluated:
    """One complete model request was assessed before provider dispatch."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.CONTEXT_PRESSURE_EVALUATED
    turn: int
    decision: CompactionDecision
    preparation: CompactionPreparation | None = None

    def __post_init__(self) -> None:
        if self.turn <= 0:
            raise ValueError("turn must be greater than zero")
        if self.preparation is not None and not self.decision.should_compact:
            raise ValueError("compaction preparation requires a positive decision")


@dataclass(frozen=True, slots=True)
class CompactionStarted:
    """One compaction operation began summary generation."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.COMPACTION_STARTED
    request: CompactionRequest


@dataclass(frozen=True, slots=True)
class CompactionCompleted:
    """One compaction completed or safely skipped."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.COMPACTION_COMPLETED
    result: CompactionResult

    def __post_init__(self) -> None:
        if self.result.status not in {
            CompactionStatus.COMPLETED,
            CompactionStatus.SKIPPED,
        }:
            raise ValueError("completed event requires completed or skipped result")


@dataclass(frozen=True, slots=True)
class CompactionFailed:
    """One compaction failed or was cancelled before mutation."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.COMPACTION_FAILED
    result: CompactionResult

    def __post_init__(self) -> None:
        if self.result.status not in {
            CompactionStatus.FAILED,
            CompactionStatus.CANCELLED,
        }:
            raise ValueError("failed event requires failed or cancelled result")


@dataclass(frozen=True, slots=True)
class MessageCompleted:
    """One complete provider-neutral assistant message was accepted."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.MESSAGE_COMPLETED
    turn: int
    message: AssistantMessage
    usage: ModelUsage | None = None


@dataclass(frozen=True, slots=True)
class AssistantTextDelta:
    """One provisional assistant text fragment for the active turn."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.ASSISTANT_TEXT_DELTA
    turn: int
    delta: str

    def __post_init__(self) -> None:
        if self.turn <= 0:
            raise ValueError("turn must be greater than zero")
        if not self.delta:
            raise ValueError("assistant text delta must not be empty")


@dataclass(frozen=True, slots=True)
class AssistantThinkingDelta:
    """One provisional reasoning fragment for the active turn."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.ASSISTANT_THINKING_DELTA
    turn: int
    delta: str

    def __post_init__(self) -> None:
        if self.turn <= 0:
            raise ValueError("turn must be greater than zero")
        if not self.delta:
            raise ValueError("assistant thinking delta must not be empty")


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """Execution of one normalized model tool call started."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.TOOL_STARTED
    turn: int
    tool_call: ModelToolCall


@dataclass(frozen=True, slots=True)
class ToolProgressed:
    """One provisional update from the active tool execution."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.TOOL_PROGRESSED
    turn: int
    tool_call: ModelToolCall
    update: ToolProgressUpdate

    def __post_init__(self) -> None:
        if self.turn <= 0:
            raise ValueError("turn must be greater than zero")


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    """Execution of one normalized model tool call settled."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.TOOL_COMPLETED
    turn: int
    tool_call: ModelToolCall
    result: ToolCallResult


@dataclass(frozen=True, slots=True)
class TurnCompleted:
    """One provider turn and its requested tool calls settled."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.TURN_COMPLETED
    turn: int


@dataclass(frozen=True, slots=True)
class AgentFinished:
    """One agent run reached its structured terminal result."""

    kind: ClassVar[AgentEventKind] = AgentEventKind.AGENT_FINISHED
    result: AgentRunResult


AgentEventPayload: TypeAlias = (
    AgentStarted
    | TurnStarted
    | SteeringApplied
    | SteeringDiscarded
    | ContextPressureEvaluated
    | CompactionStarted
    | CompactionCompleted
    | CompactionFailed
    | MessageCompleted
    | AssistantTextDelta
    | AssistantThinkingDelta
    | ToolStarted
    | ToolProgressed
    | ToolCompleted
    | TurnCompleted
    | AgentFinished
)


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Immutable event envelope shared by every lifecycle payload."""

    agent_id: str
    run_id: str
    sequence: int
    payload: AgentEventPayload

    @property
    def kind(self) -> AgentEventKind:
        return self.payload.kind


class AgentEventSink(Protocol):
    """Read-only observer receiving ordered events for an agent run."""

    async def emit(self, event: AgentEvent) -> None:
        """Observe an event without changing agent behavior."""


class AgentEventSinkError(RuntimeError):
    """Raised after one or more sinks fail during event fan-out."""


class CompositeAgentEventSink:
    """Forward each event to multiple ordered read-only observers."""

    def __init__(self, sinks: Iterable[AgentEventSink]) -> None:
        self.sinks = tuple(sinks)

    async def emit(self, event: AgentEvent) -> None:
        errors: list[Exception] = []
        for sink in self.sinks:
            try:
                await sink.emit(event)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise AgentEventSinkError(
                f"{len(errors)} agent event sink(s) failed"
            ) from errors[0]


class AgentEventEmitter:
    """Create ordered event envelopes and isolate observer failures."""

    def __init__(
        self,
        *,
        agent_id: str,
        sink: AgentEventSink | None,
        logger: logging.Logger,
    ) -> None:
        self.agent_id = agent_id
        self.sink = sink
        self.logger = logger
        self._run_id: str | None = None
        self._sequence = 0

    def begin_run(self) -> str:
        """Open a new event sequence and return its correlation id."""

        if self._run_id is not None:
            raise RuntimeError("an agent event run is already active")
        self._run_id = uuid4().hex
        self._sequence = 0
        return self._run_id

    def end_run(self, run_id: str) -> None:
        """Close the active event sequence."""

        if self._run_id != run_id:
            raise RuntimeError("agent event run id does not match active run")
        self._run_id = None
        self._sequence = 0

    async def emit(self, payload: AgentEventPayload) -> AgentEvent:
        """Publish one event without allowing Sink failures to fail the run."""

        if self._run_id is None:
            raise RuntimeError("agent event run is not active")

        self._sequence += 1
        event = AgentEvent(
            agent_id=self.agent_id,
            run_id=self._run_id,
            sequence=self._sequence,
            payload=payload,
        )
        if self.sink is None:
            return event

        try:
            await self.sink.emit(event)
        except Exception as exc:
            self.logger.warning(
                "Agent event sink failed for %s: %s",
                event.kind,
                exc,
            )
        return event
