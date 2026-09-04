"""Host-owned Checkpoint capture at Runtime-safe semantic boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from ejagent._trajectory.context import (
    TrajectoryContextEvent,
    TrajectoryContextEventKind,
    TrajectoryContextFrame,
)
from ejagent._trajectory.shadow import (
    EnvironmentFact,
    ShadowTrajectoryAnalyzer,
    TrajectoryAssessment,
    TrajectoryCheckpoint,
    TrajectoryVerdict,
)
from ejagent.contracts.control import CancellationToken
from ejagent.contracts.json import JsonObject, freeze_json_object
from ejagent.kernel.trajectory import (
    CheckpointSignal,
    CheckpointTrigger,
)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must not be negative")
    return value


def _freeze_verdicts(
    values: Mapping[str, bool | None],
    label: str,
    *,
    allow_empty: bool = False,
) -> Mapping[str, bool | None]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    frozen: dict[str, bool | None] = {}
    for name, verdict in values.items():
        _required_text(name, f"{label} key")
        if verdict is not None and not isinstance(verdict, bool):
            raise TypeError(f"{label}[{name!r}] must be bool or None")
        frozen[name] = verdict
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class CheckpointEvaluationRequest:
    """Evaluator input with the host-assigned Checkpoint identity."""

    checkpoint_id: str
    signal: CheckpointSignal
    previous_checkpoint: TrajectoryCheckpoint | None

    def __post_init__(self) -> None:
        _required_text(self.checkpoint_id, "evaluation checkpoint_id")
        if not isinstance(self.signal, CheckpointSignal):
            raise TypeError("evaluation signal must be a CheckpointSignal")
        if self.previous_checkpoint is not None and not isinstance(
            self.previous_checkpoint, TrajectoryCheckpoint
        ):
            raise TypeError(
                "previous_checkpoint must be a TrajectoryCheckpoint or None"
            )


@dataclass(frozen=True, slots=True)
class CheckpointEvaluation:
    """Fresh, goal-relative Facts returned by a host-owned Evaluator."""

    projection_version: str
    state_fingerprint: str
    environment_facts: JsonObject
    requirements: Mapping[str, bool | None]
    constraints: Mapping[str, bool | None]
    new_evidence: tuple[str, ...] = ()
    facts: tuple[EnvironmentFact, ...] = ()
    fact_capture_complete: bool = False

    def __post_init__(self) -> None:
        _required_text(self.projection_version, "evaluation projection_version")
        _required_text(self.state_fingerprint, "evaluation state_fingerprint")
        object.__setattr__(
            self,
            "environment_facts",
            freeze_json_object(
                self.environment_facts,
                label="evaluation environment_facts",
            ),
        )
        object.__setattr__(
            self,
            "requirements",
            _freeze_verdicts(self.requirements, "evaluation requirements"),
        )
        object.__setattr__(
            self,
            "constraints",
            _freeze_verdicts(
                self.constraints,
                "evaluation constraints",
                allow_empty=True,
            ),
        )
        evidence = tuple(self.new_evidence)
        for item in evidence:
            _required_text(item, "evaluation new_evidence item")
        object.__setattr__(self, "new_evidence", evidence)
        facts = tuple(self.facts)
        if not all(isinstance(item, EnvironmentFact) for item in facts):
            raise TypeError("evaluation facts must contain EnvironmentFact values")
        object.__setattr__(self, "facts", facts)
        if not isinstance(self.fact_capture_complete, bool):
            raise TypeError("evaluation fact_capture_complete must be a bool")


class CheckpointEvaluator(Protocol):
    """Host boundary that turns current environment truth into an evaluation."""

    async def evaluate(
        self,
        request: CheckpointEvaluationRequest,
        *,
        cancellation: CancellationToken,
    ) -> CheckpointEvaluation:
        """Observe current truth without relying on Actor narration."""


class CheckpointProtocolError(RuntimeError):
    """The Runtime or host Evaluator violated the Checkpoint protocol."""


@dataclass(frozen=True, slots=True)
class TrajectoryUpdate:
    """One captured Checkpoint and its immediately available Assessment."""

    signal: CheckpointSignal
    checkpoint: TrajectoryCheckpoint
    assessment: TrajectoryAssessment
    context_event: TrajectoryContextEvent
    completion_allowed: bool | None

    @property
    def checkpoint_id(self) -> str:
        """Expose the stable Runtime receipt field."""

        return self.checkpoint.checkpoint_id

    @property
    def verdict(self) -> str:
        """Expose the stable Runtime receipt field."""

        return self.assessment.verdict.value

    def to_context_frame(
        self,
        *,
        goal: str,
        next_turn: int,
        current_plan: str | None = None,
        refuted_hypotheses: tuple[str, ...] = (),
    ) -> TrajectoryContextFrame:
        """Project this update only at the caller-declared next decision boundary."""

        return TrajectoryContextFrame(
            run_id=self.signal.run_id,
            turn=next_turn,
            goal=goal,
            checkpoint=self.checkpoint,
            progress=self.assessment.progress[-1],
            event=self.context_event,
            current_plan=current_plan,
            refuted_hypotheses=refuted_hypotheses,
        )


TrajectoryUpdateSink = Callable[[TrajectoryUpdate], None]
TrajectoryRunCloseSink = Callable[[str, tuple[TrajectoryCheckpoint, ...]], None]


@dataclass(slots=True)
class _RunState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    checkpoints: list[TrajectoryCheckpoint] = field(default_factory=list)


class OnlineTrajectoryMonitor:
    """Serialize capture per Run and assess each new Checkpoint online."""

    def __init__(
        self,
        evaluator: CheckpointEvaluator,
        *,
        analyzer: ShadowTrajectoryAnalyzer | None = None,
        update_sink: TrajectoryUpdateSink | None = None,
        run_close_sink: TrajectoryRunCloseSink | None = None,
    ) -> None:
        if update_sink is not None and not callable(update_sink):
            raise TypeError("update_sink must be callable or None")
        if run_close_sink is not None and not callable(run_close_sink):
            raise TypeError("run_close_sink must be callable or None")
        self._evaluator = evaluator
        self._analyzer = analyzer or ShadowTrajectoryAnalyzer()
        self._update_sink = update_sink
        self._run_close_sink = run_close_sink
        self._runs: dict[str, _RunState] = {}

    async def capture(
        self,
        signal: CheckpointSignal,
        *,
        cancellation: CancellationToken,
    ) -> TrajectoryUpdate:
        if not isinstance(signal, CheckpointSignal):
            raise TypeError("signal must be a CheckpointSignal")
        if not isinstance(cancellation, CancellationToken):
            raise TypeError("cancellation must be a CancellationToken")
        cancellation.raise_if_cancelled()
        state = self._runs.setdefault(signal.run_id, _RunState())
        await cancellation.run(state.lock.acquire())
        try:
            previous = state.checkpoints[-1] if state.checkpoints else None
            self._validate_transition(signal, previous)
            checkpoint_id = f"{signal.run_id}:cp{len(state.checkpoints)}"
            request = CheckpointEvaluationRequest(
                checkpoint_id=checkpoint_id,
                signal=signal,
                previous_checkpoint=previous,
            )
            evaluation = await cancellation.run(
                self._evaluator.evaluate(request, cancellation=cancellation)
            )
            if not isinstance(evaluation, CheckpointEvaluation):
                raise CheckpointProtocolError(
                    "CheckpointEvaluator.evaluate() must return CheckpointEvaluation"
                )
            if any(item.checkpoint_id != checkpoint_id for item in evaluation.facts):
                raise CheckpointProtocolError(
                    "Evaluator Facts must reference the assigned Checkpoint"
                )
            checkpoint = TrajectoryCheckpoint(
                checkpoint_id=checkpoint_id,
                projection_version=evaluation.projection_version,
                state_fingerprint=evaluation.state_fingerprint,
                environment_facts=evaluation.environment_facts,
                requirements=evaluation.requirements,
                constraints=evaluation.constraints,
                new_evidence=evaluation.new_evidence,
                actor_action_count=signal.cumulative_cost.actor_actions,
                causal_action_signatures=tuple(
                    item.signature for item in signal.causal_actions
                ),
                causally_complete=signal.causally_complete,
                facts=evaluation.facts,
                fact_capture_complete=evaluation.fact_capture_complete,
                causal_batch_id=signal.causal_batch_id,
                unattributed_action_ids=signal.unattributed_action_ids,
                causal_exclusion_reason=signal.causal_exclusion_reason,
                cumulative_cost=signal.cumulative_cost,
                turn=signal.turn,
                capture_trigger=signal.trigger.value,
            )
            candidate_history = (*state.checkpoints, checkpoint)
            assessment = self._analyzer.assess(candidate_history)
            context_event = self._context_event(signal, checkpoint, assessment)
            completion_allowed = self._completion_allowed(
                signal,
                checkpoint,
                assessment,
            )
            update = TrajectoryUpdate(
                signal,
                checkpoint,
                assessment,
                context_event,
                completion_allowed,
            )
            if self._update_sink is not None:
                self._update_sink(update)
            state.checkpoints.append(checkpoint)
            return update
        finally:
            state.lock.release()

    def checkpoints(self, run_id: str) -> tuple[TrajectoryCheckpoint, ...]:
        """Return a snapshot without ending the Run."""

        _required_text(run_id, "run_id")
        state = self._runs.get(run_id)
        return () if state is None else tuple(state.checkpoints)

    def close_run(self, run_id: str) -> tuple[TrajectoryCheckpoint, ...]:
        _required_text(run_id, "run_id")
        state = self._runs.pop(run_id, None)
        if state is None:
            if self._run_close_sink is not None:
                self._run_close_sink(run_id, ())
            return ()
        if state.lock.locked():
            self._runs[run_id] = state
            raise CheckpointProtocolError(
                "cannot close a Run while Checkpoint capture is active"
            )
        checkpoints = tuple(state.checkpoints)
        if self._run_close_sink is not None:
            self._run_close_sink(run_id, checkpoints)
        return checkpoints

    @staticmethod
    def _validate_transition(
        signal: CheckpointSignal,
        previous: TrajectoryCheckpoint | None,
    ) -> None:
        if previous is None:
            if signal.trigger is not CheckpointTrigger.BASELINE:
                raise CheckpointProtocolError(
                    "the first Checkpoint signal for a Run must be baseline"
                )
            return
        if signal.trigger is CheckpointTrigger.BASELINE:
            raise CheckpointProtocolError("a Run may have only one baseline")
        previous_cost = previous.cumulative_cost
        assert previous_cost is not None
        if signal.turn < 1:
            raise CheckpointProtocolError("post-baseline signal turn must be positive")
        if signal.turn < previous.turn:
            raise CheckpointProtocolError("Checkpoint signal turn regressed")
        expected_action_count = previous_cost.actor_actions + len(signal.causal_actions)
        if signal.cumulative_cost.actor_actions != expected_action_count:
            raise CheckpointProtocolError(
                "cumulative actor Action cost does not match attributed Actions"
            )
        for name in ("model_requests", "total_tokens", "elapsed_ms"):
            old_value = getattr(previous_cost, name)
            new_value = getattr(signal.cumulative_cost, name)
            if old_value is not None and new_value is None:
                raise CheckpointProtocolError(f"cumulative {name} became unavailable")
            if (
                old_value is not None
                and new_value is not None
                and new_value < old_value
            ):
                raise CheckpointProtocolError(f"cumulative {name} regressed")

    @staticmethod
    def _completion_allowed(
        signal: CheckpointSignal,
        checkpoint: TrajectoryCheckpoint,
        assessment: TrajectoryAssessment,
    ) -> bool | None:
        if signal.trigger is not CheckpointTrigger.COMPLETION_PROPOSED:
            return None
        all_items_satisfied = all(
            value is True
            for value in (
                *checkpoint.requirements.values(),
                *checkpoint.constraints.values(),
            )
        )
        return (
            all_items_satisfied
            and checkpoint.fact_capture_complete
            and assessment.verdict is not TrajectoryVerdict.INSUFFICIENT_EVIDENCE
        )

    @staticmethod
    def _context_event(
        signal: CheckpointSignal,
        checkpoint: TrajectoryCheckpoint,
        assessment: TrajectoryAssessment,
    ) -> TrajectoryContextEvent:
        progress = assessment.progress[-1]
        affected = tuple(
            dict.fromkeys(
                (
                    *progress.gained_requirements,
                    *progress.regressed_requirements,
                    *progress.newly_violated_constraints,
                    *progress.recovered_constraints,
                )
            )
        )
        evidence_refs = tuple(
            dict.fromkeys(
                (
                    *(item.evidence_ref for item in checkpoint.current_facts),
                    *checkpoint.new_evidence,
                )
            )
        )
        event_arguments: dict[str, tuple[str, ...]] = {}
        if signal.trigger is CheckpointTrigger.COMPLETION_PROPOSED and not all(
            value is True
            for value in (
                *checkpoint.requirements.values(),
                *checkpoint.constraints.values(),
            )
        ):
            kind = TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED
            event_arguments["affected_items"] = tuple(
                name
                for name, value in (
                    *checkpoint.requirements.items(),
                    *checkpoint.constraints.items(),
                )
                if value is not True
            )
            event_arguments["missing_evidence"] = tuple(
                name
                for name, value in (
                    *checkpoint.requirements.items(),
                    *checkpoint.constraints.items(),
                )
                if value is None
            )
        elif (
            signal.trigger is CheckpointTrigger.COMPLETION_PROPOSED
            and not checkpoint.fact_capture_complete
        ):
            kind = TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED
            event_arguments["missing_evidence"] = (
                "complete current Environment Fact capture",
            )
        elif assessment.verdict is TrajectoryVerdict.NON_PROGRESS_CYCLE:
            kind = TrajectoryContextEventKind.CYCLE_CONFIRMED
            event_arguments["causal_actions"] = assessment.repeated_action_signatures
        elif progress.newly_violated_constraints:
            kind = TrajectoryContextEventKind.CONSTRAINT_VIOLATED
            event_arguments["affected_items"] = progress.newly_violated_constraints
        elif signal.trigger is CheckpointTrigger.EXTERNAL_CHANGE:
            invalidated = tuple(item.fact_id for item in checkpoint.invalidated_facts)
            if not invalidated:
                raise CheckpointProtocolError(
                    "external change evaluation must explicitly invalidate prior Facts"
                )
            kind = TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED
            event_arguments["invalidated_fact_ids"] = invalidated
        elif assessment.verdict is TrajectoryVerdict.CYCLE_SUSPECTED:
            kind = TrajectoryContextEventKind.CYCLE_SUSPECTED
        elif affected or checkpoint.new_evidence:
            kind = TrajectoryContextEventKind.PROGRESS_EVALUATED
            event_arguments["affected_items"] = affected
        else:
            kind = TrajectoryContextEventKind.FACTS_UPDATED
        return TrajectoryContextEvent(
            event_id=f"{checkpoint.checkpoint_id}:event",
            kind=kind,
            evidence_refs=evidence_refs,
            **event_arguments,
        )
