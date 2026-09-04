from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from ejagent.contracts.audit import RunAudit
from ejagent.contracts.json import (
    JsonObject,
    JsonValue,
    freeze_json_object,
    freeze_json_value,
)
from ejagent.contracts.runs import AuditRecord, RunStatus, StopReason
from ejagent.kernel.trajectory import TrajectoryCost


class TrajectoryVerdict(StrEnum):
    """Observation-only assessment; it has no Runtime control authority."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CAUSALLY_AMBIGUOUS = "causally_ambiguous"
    NO_CYCLE = "no_cycle"
    CYCLE_SUSPECTED = "cycle_suspected"
    NON_PROGRESS_CYCLE = "non_progress_cycle"


class ProgressStatus(StrEnum):
    """Constraint-aware interpretation of one Progress Delta."""

    UNCHANGED = "unchanged"
    EVIDENCE_GAINED = "evidence_gained"
    ADVANCED = "advanced"
    BLOCKED = "blocked"
    REGRESSED = "regressed"


class FactValidity(StrEnum):
    """Validity of one immutable Environment Fact at a checkpoint."""

    CURRENT = "current"
    INVALIDATED = "invalidated"
    STALE = "stale"
    UNKNOWN = "unknown"


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
class EnvironmentFact:
    """One source-attributed assertion and its checkpoint-local validity."""

    fact_id: str
    subject: str
    predicate: str
    value: JsonValue
    scope: tuple[str, ...]
    source: str
    observed_at: datetime
    checkpoint_id: str
    evidence_ref: str
    freshness: str
    authority: str
    validity: FactValidity = FactValidity.CURRENT
    invalidated_at_checkpoint: str | None = None
    validity_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.fact_id, "fact_id")
        _required_text(self.subject, "fact subject")
        _required_text(self.predicate, "fact predicate")
        object.__setattr__(
            self,
            "value",
            freeze_json_value(self.value, label="fact value"),
        )
        scope = tuple(self.scope)
        if not scope:
            raise ValueError("fact scope must not be empty")
        for item in scope:
            _required_text(item, "fact scope item")
        object.__setattr__(self, "scope", scope)
        _required_text(self.source, "fact source")
        if not isinstance(self.observed_at, datetime):
            raise TypeError("fact observed_at must be a datetime")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("fact observed_at must be timezone-aware")
        _required_text(self.checkpoint_id, "fact checkpoint_id")
        _required_text(self.evidence_ref, "fact evidence_ref")
        _required_text(self.freshness, "fact freshness")
        _required_text(self.authority, "fact authority")
        if not isinstance(self.validity, FactValidity):
            raise TypeError("fact validity must be a FactValidity")
        if self.validity is FactValidity.CURRENT:
            if self.invalidated_at_checkpoint is not None:
                raise ValueError("current Fact cannot have invalidated_at_checkpoint")
            if self.validity_reason is not None:
                raise ValueError("current Fact cannot have validity_reason")
            return
        if self.validity_reason is None:
            raise ValueError("non-current Fact must have validity_reason")
        _required_text(self.validity_reason, "fact validity_reason")
        if self.validity is FactValidity.INVALIDATED:
            if self.invalidated_at_checkpoint is None:
                raise ValueError("invalidated Fact must have invalidated_at_checkpoint")
            _required_text(
                self.invalidated_at_checkpoint,
                "fact invalidated_at_checkpoint",
            )
        elif self.invalidated_at_checkpoint is not None:
            raise ValueError(
                "only an invalidated Fact may have invalidated_at_checkpoint"
            )

    @property
    def comparison_key(self) -> tuple[object, ...]:
        """Return current truth fields used for State equivalence."""

        return (
            self.subject,
            self.predicate,
            self.value,
            self.scope,
            self.source,
            self.evidence_ref,
            self.freshness,
            self.authority,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryCheckpoint:
    """Host-owned Facts captured at one causally identified environment State."""

    checkpoint_id: str
    projection_version: str
    state_fingerprint: str
    environment_facts: JsonObject
    requirements: Mapping[str, bool | None]
    constraints: Mapping[str, bool | None]
    new_evidence: tuple[str, ...] = ()
    actor_action_count: int = 0
    causal_action_signatures: tuple[str, ...] = ()
    causally_complete: bool = True
    facts: tuple[EnvironmentFact, ...] = ()
    fact_capture_complete: bool = False
    causal_batch_id: str | None = None
    unattributed_action_ids: tuple[str, ...] = ()
    causal_exclusion_reason: str | None = None
    cumulative_cost: TrajectoryCost | None = None
    turn: int = 0
    capture_trigger: str = "legacy"

    def __post_init__(self) -> None:
        _required_text(self.checkpoint_id, "checkpoint_id")
        _required_text(self.projection_version, "projection_version")
        _required_text(self.state_fingerprint, "state_fingerprint")
        _non_negative_integer(self.turn, "checkpoint turn")
        _required_text(self.capture_trigger, "capture_trigger")
        object.__setattr__(
            self,
            "environment_facts",
            freeze_json_object(self.environment_facts, label="environment_facts"),
        )
        object.__setattr__(
            self,
            "requirements",
            _freeze_verdicts(self.requirements, "requirements"),
        )
        object.__setattr__(
            self,
            "constraints",
            _freeze_verdicts(self.constraints, "constraints", allow_empty=True),
        )
        evidence = tuple(self.new_evidence)
        for item in evidence:
            _required_text(item, "new_evidence item")
        object.__setattr__(self, "new_evidence", evidence)
        _non_negative_integer(self.actor_action_count, "actor_action_count")
        signatures = tuple(self.causal_action_signatures)
        for item in signatures:
            _required_text(item, "causal_action_signatures item")
        object.__setattr__(self, "causal_action_signatures", signatures)
        if not isinstance(self.causally_complete, bool):
            raise TypeError("causally_complete must be a bool")
        facts = tuple(self.facts)
        if not all(isinstance(item, EnvironmentFact) for item in facts):
            raise TypeError("facts must contain EnvironmentFact values")
        fact_ids = [item.fact_id for item in facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("facts must have unique fact_id values")
        current_claims = [
            (item.subject, item.predicate, item.scope)
            for item in facts
            if item.validity is FactValidity.CURRENT
        ]
        if len(current_claims) != len(set(current_claims)):
            raise ValueError(
                "current Facts must not assert the same scoped claim twice"
            )
        object.__setattr__(self, "facts", facts)
        if not isinstance(self.fact_capture_complete, bool):
            raise TypeError("fact_capture_complete must be a bool")
        if self.fact_capture_complete and not facts:
            raise ValueError("complete fact capture must contain at least one Fact")
        if self.causal_batch_id is not None:
            _required_text(self.causal_batch_id, "causal_batch_id")
        unattributed = tuple(self.unattributed_action_ids)
        for item in unattributed:
            _required_text(item, "unattributed_action_ids item")
        object.__setattr__(self, "unattributed_action_ids", unattributed)
        if self.causally_complete:
            if unattributed:
                raise ValueError(
                    "causally complete checkpoint cannot have unattributed Actions"
                )
            if self.causal_exclusion_reason is not None:
                raise ValueError(
                    "causally complete checkpoint cannot have an exclusion reason"
                )
        else:
            if self.causal_exclusion_reason is None:
                raise ValueError(
                    "causally incomplete checkpoint must have an exclusion reason"
                )
            _required_text(
                self.causal_exclusion_reason,
                "causal_exclusion_reason",
            )
        cost = self.cumulative_cost
        if cost is None:
            cost = TrajectoryCost(actor_actions=self.actor_action_count)
            object.__setattr__(self, "cumulative_cost", cost)
        elif not isinstance(cost, TrajectoryCost):
            raise TypeError("cumulative_cost must be a TrajectoryCost or None")
        elif cost.actor_actions != self.actor_action_count:
            raise ValueError(
                "cumulative_cost actor_actions must match actor_action_count"
            )

    @property
    def current_facts(self) -> tuple[EnvironmentFact, ...]:
        return tuple(
            item for item in self.facts if item.validity is FactValidity.CURRENT
        )

    @property
    def invalidated_facts(self) -> tuple[EnvironmentFact, ...]:
        return tuple(
            item for item in self.facts if item.validity is FactValidity.INVALIDATED
        )


@dataclass(frozen=True, slots=True)
class NormalizedAction:
    """One model-proposed Tool Action reconstructed from RunAudit."""

    tool_call_id: str
    turn: int
    tool_name: str
    arguments: JsonObject
    proposed_audit_sequence: int | None
    started_audit_sequence: int | None
    completed_audit_sequence: int | None
    is_error: bool | None

    def __post_init__(self) -> None:
        _required_text(self.tool_call_id, "tool_call_id")
        _non_negative_integer(self.turn, "turn")
        _required_text(self.tool_name, "tool_name")
        object.__setattr__(
            self,
            "arguments",
            freeze_json_object(self.arguments, label="action arguments"),
        )


@dataclass(frozen=True, slots=True)
class NormalizedObservation:
    """One Tool completion in actual Audit completion order."""

    tool_call_id: str
    turn: int
    tool_name: str
    result: JsonValue
    control: str
    is_error: bool
    completed_audit_sequence: int

    def __post_init__(self) -> None:
        _required_text(self.tool_call_id, "tool_call_id")
        _non_negative_integer(self.turn, "turn")
        _required_text(self.tool_name, "tool_name")
        object.__setattr__(
            self,
            "result",
            freeze_json_value(self.result, label="observation result"),
        )
        _required_text(self.control, "control")
        if not isinstance(self.is_error, bool):
            raise TypeError("is_error must be a bool")
        _non_negative_integer(self.completed_audit_sequence, "completed_audit_sequence")


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Goal-relative assessment derived from one complete checkpoint."""

    checkpoint_id: str
    requirements: Mapping[str, bool | None]
    constraints: Mapping[str, bool | None]
    current_requirement_coverage: float
    best_requirement_coverage: float
    requirement_coverage_delta: float
    task_progress_delta: float | None
    status: ProgressStatus
    gained_requirements: tuple[str, ...]
    regressed_requirements: tuple[str, ...]
    violated_constraints: tuple[str, ...]
    unresolved_constraints: tuple[str, ...]
    newly_violated_constraints: tuple[str, ...]
    recovered_constraints: tuple[str, ...]
    new_evidence: tuple[str, ...]
    actor_actions_since_previous: int
    cost_since_previous: TrajectoryCost

    def __post_init__(self) -> None:
        _required_text(self.checkpoint_id, "checkpoint_id")
        object.__setattr__(
            self,
            "requirements",
            _freeze_verdicts(self.requirements, "progress requirements"),
        )
        object.__setattr__(
            self,
            "constraints",
            _freeze_verdicts(
                self.constraints,
                "progress constraints",
                allow_empty=True,
            ),
        )
        if not isinstance(self.status, ProgressStatus):
            raise TypeError("progress status must be a ProgressStatus")
        if not isinstance(self.cost_since_previous, TrajectoryCost):
            raise TypeError("cost_since_previous must be a TrajectoryCost")


@dataclass(frozen=True, slots=True)
class TrajectoryAssessment:
    """Online assessment derived only from immutable Checkpoints."""

    verdict: TrajectoryVerdict
    period: int | None
    candidate_checkpoint_ids: tuple[str, ...]
    repeated_action_path: bool
    repeated_action_signatures: tuple[str, ...]
    task_progress_over_repeated_window: float | None
    progress: tuple[ProgressSnapshot, ...]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrajectoryReport:
    """Detached Shadow Mode output; consumers cannot affect the observed Run."""

    run_id: str
    run_status: RunStatus
    stop_reason: StopReason
    committed: bool
    base_revision: int
    resulting_revision: int
    turns: int
    total_tokens: int
    request_count: int
    verdict: TrajectoryVerdict
    period: int | None
    candidate_checkpoint_ids: tuple[str, ...]
    repeated_action_path: bool
    repeated_action_signatures: tuple[str, ...]
    task_progress_over_repeated_window: float | None
    actions: tuple[NormalizedAction, ...]
    observations: tuple[NormalizedObservation, ...]
    progress: tuple[ProgressSnapshot, ...]
    diagnostics: tuple[str, ...] = ()


@dataclass(slots=True)
class _ActionBuilder:
    order: int
    tool_call_id: str
    turn: int
    tool_name: str
    arguments: JsonObject = field(default_factory=dict)
    proposed_sequence: int | None = None
    started_sequence: int | None = None
    completed_sequence: int | None = None
    is_error: bool | None = None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _json_object(value: object) -> JsonObject | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return freeze_json_object(value, label="audit arguments")
    except (TypeError, ValueError):
        return None


def _action_builder(
    builders: dict[str, _ActionBuilder],
    *,
    call_id: str,
    turn: int,
    tool_name: str,
    arguments: JsonObject | None = None,
) -> _ActionBuilder:
    existing = builders.get(call_id)
    if existing is not None:
        return existing
    builder = _ActionBuilder(
        order=len(builders),
        tool_call_id=call_id,
        turn=turn,
        tool_name=tool_name,
        arguments=arguments or {},
    )
    builders[call_id] = builder
    return builder


def _normalize_audit(
    records: tuple[AuditRecord, ...],
) -> tuple[
    tuple[NormalizedAction, ...],
    tuple[NormalizedObservation, ...],
    tuple[str, ...],
]:
    builders: dict[str, _ActionBuilder] = {}
    observations: list[NormalizedObservation] = []
    diagnostics: list[str] = []
    for record in records:
        payload = record.payload
        if record.kind == "assistant_message":
            turn = _integer(payload.get("turn"))
            calls = payload.get("tool_calls")
            if turn is None or not isinstance(calls, tuple):
                diagnostics.append(f"audit:{record.sequence}:invalid assistant_message")
                continue
            for raw_call in calls:
                call = _mapping(raw_call)
                if call is None:
                    diagnostics.append(f"audit:{record.sequence}:invalid tool proposal")
                    continue
                call_id = call.get("id")
                tool_name = call.get("name")
                arguments = _json_object(call.get("arguments"))
                if not isinstance(call_id, str) or not isinstance(tool_name, str):
                    diagnostics.append(f"audit:{record.sequence}:invalid tool proposal")
                    continue
                builder = _action_builder(
                    builders,
                    call_id=call_id,
                    turn=turn,
                    tool_name=tool_name,
                    arguments=arguments,
                )
                builder.proposed_sequence = record.sequence
        elif record.kind == "tool_started":
            turn = _integer(payload.get("turn"))
            call_id = payload.get("tool_call_id")
            tool_name = payload.get("tool_name")
            arguments = _json_object(payload.get("arguments"))
            if (
                turn is None
                or not isinstance(call_id, str)
                or not isinstance(tool_name, str)
            ):
                diagnostics.append(f"audit:{record.sequence}:invalid tool_started")
                continue
            builder = _action_builder(
                builders,
                call_id=call_id,
                turn=turn,
                tool_name=tool_name,
                arguments=arguments,
            )
            builder.started_sequence = record.sequence
        elif record.kind == "tool_completed":
            turn = _integer(payload.get("turn"))
            call_id = payload.get("tool_call_id")
            tool_name = payload.get("tool_name")
            control = payload.get("control")
            is_error = payload.get("is_error")
            if (
                turn is None
                or not isinstance(call_id, str)
                or not isinstance(tool_name, str)
                or not isinstance(control, str)
                or not isinstance(is_error, bool)
            ):
                diagnostics.append(f"audit:{record.sequence}:invalid tool_completed")
                continue
            builder = _action_builder(
                builders,
                call_id=call_id,
                turn=turn,
                tool_name=tool_name,
            )
            builder.completed_sequence = record.sequence
            builder.is_error = is_error
            observations.append(
                NormalizedObservation(
                    tool_call_id=call_id,
                    turn=turn,
                    tool_name=tool_name,
                    result=payload.get("result"),
                    control=control,
                    is_error=is_error,
                    completed_audit_sequence=record.sequence,
                )
            )
    actions = tuple(
        NormalizedAction(
            tool_call_id=builder.tool_call_id,
            turn=builder.turn,
            tool_name=builder.tool_name,
            arguments=builder.arguments,
            proposed_audit_sequence=builder.proposed_sequence,
            started_audit_sequence=builder.started_sequence,
            completed_audit_sequence=builder.completed_sequence,
            is_error=builder.is_error,
        )
        for builder in sorted(builders.values(), key=lambda item: item.order)
    )
    return actions, tuple(observations), tuple(diagnostics)


def _coverage(requirements: Mapping[str, bool | None]) -> float:
    return sum(value is True for value in requirements.values()) / len(requirements)


def _cost_delta(
    current: TrajectoryCost,
    previous: TrajectoryCost,
    *,
    checkpoint_id: str,
    diagnostics: list[str],
) -> TrajectoryCost:
    def delta(name: str, value: int | None, prior: int | None) -> int | None:
        if value is None:
            return None
        if prior is None:
            return value
        difference = value - prior
        if difference < 0:
            diagnostics.append(
                f"checkpoint:{checkpoint_id}:cumulative {name} regressed"
            )
            return 0
        return difference

    return TrajectoryCost(
        actor_actions=delta(
            "actor action count",
            current.actor_actions,
            previous.actor_actions,
        )
        or 0,
        model_requests=delta(
            "model request count",
            current.model_requests,
            previous.model_requests,
        )
        or 0,
        total_tokens=delta(
            "total token count",
            current.total_tokens,
            previous.total_tokens,
        ),
        elapsed_ms=delta(
            "elapsed time",
            current.elapsed_ms,
            previous.elapsed_ms,
        ),
    )


def _progress(
    checkpoints: tuple[TrajectoryCheckpoint, ...],
) -> tuple[tuple[ProgressSnapshot, ...], tuple[str, ...]]:
    snapshots: list[ProgressSnapshot] = []
    diagnostics: list[str] = []
    expected_requirements = tuple(checkpoints[0].requirements)
    expected_constraints = tuple(checkpoints[0].constraints)
    previous_requirements: Mapping[str, bool | None] | None = None
    previous_constraints: Mapping[str, bool | None] | None = None
    previous_coverage: float | None = None
    previous_cost = TrajectoryCost()
    best_coverage = 0.0
    for checkpoint in checkpoints:
        if tuple(checkpoint.requirements) != expected_requirements:
            diagnostics.append(
                f"checkpoint:{checkpoint.checkpoint_id}:requirement schema changed"
            )
        if tuple(checkpoint.constraints) != expected_constraints:
            diagnostics.append(
                f"checkpoint:{checkpoint.checkpoint_id}:constraint schema changed"
            )
        cost = checkpoint.cumulative_cost
        assert cost is not None
        cost_delta = _cost_delta(
            cost,
            previous_cost,
            checkpoint_id=checkpoint.checkpoint_id,
            diagnostics=diagnostics,
        )
        coverage = _coverage(checkpoint.requirements)
        requirement_delta = (
            0.0 if previous_coverage is None else coverage - previous_coverage
        )
        gained: tuple[str, ...] = ()
        regressed: tuple[str, ...] = ()
        if previous_requirements is not None:
            gained = tuple(
                name
                for name, verdict in checkpoint.requirements.items()
                if verdict is True and previous_requirements.get(name) is not True
            )
            regressed = tuple(
                name
                for name, verdict in checkpoint.requirements.items()
                if verdict is not True and previous_requirements.get(name) is True
            )
        violated = tuple(
            name for name, verdict in checkpoint.constraints.items() if verdict is False
        )
        unresolved = tuple(
            name for name, verdict in checkpoint.constraints.items() if verdict is None
        )
        newly_violated: tuple[str, ...] = ()
        recovered: tuple[str, ...] = ()
        if previous_constraints is not None:
            newly_violated = tuple(
                name
                for name, verdict in checkpoint.constraints.items()
                if verdict is False and previous_constraints.get(name) is not False
            )
            recovered = tuple(
                name
                for name, verdict in checkpoint.constraints.items()
                if verdict is True and previous_constraints.get(name) is not True
            )
        if regressed or newly_violated:
            status = ProgressStatus.REGRESSED
        elif violated or unresolved:
            status = ProgressStatus.BLOCKED
        elif requirement_delta > 0:
            status = ProgressStatus.ADVANCED
        elif checkpoint.new_evidence:
            status = ProgressStatus.EVIDENCE_GAINED
        else:
            status = ProgressStatus.UNCHANGED
        best_coverage = max(best_coverage, coverage)
        snapshots.append(
            ProgressSnapshot(
                checkpoint_id=checkpoint.checkpoint_id,
                requirements=checkpoint.requirements,
                constraints=checkpoint.constraints,
                current_requirement_coverage=coverage,
                best_requirement_coverage=best_coverage,
                requirement_coverage_delta=requirement_delta,
                task_progress_delta=(
                    requirement_delta if not violated and not unresolved else None
                ),
                status=status,
                gained_requirements=gained,
                regressed_requirements=regressed,
                violated_constraints=violated,
                unresolved_constraints=unresolved,
                newly_violated_constraints=newly_violated,
                recovered_constraints=recovered,
                new_evidence=checkpoint.new_evidence,
                actor_actions_since_previous=cost_delta.actor_actions,
                cost_since_previous=cost_delta,
            )
        )
        previous_requirements = checkpoint.requirements
        previous_constraints = checkpoint.constraints
        previous_coverage = coverage
        previous_cost = cost
    return tuple(snapshots), tuple(diagnostics)


def _equivalent(left: TrajectoryCheckpoint, right: TrajectoryCheckpoint) -> bool:
    if left.facts or right.facts:

        def claim_order(item: EnvironmentFact) -> tuple[object, ...]:
            return (item.subject, item.predicate, item.scope)

        left_facts = tuple(
            item.comparison_key for item in sorted(left.current_facts, key=claim_order)
        )
        right_facts = tuple(
            item.comparison_key for item in sorted(right.current_facts, key=claim_order)
        )
        facts_equal = left_facts == right_facts
    else:
        facts_equal = left.environment_facts == right.environment_facts
    return (
        left.projection_version == right.projection_version
        and left.state_fingerprint == right.state_fingerprint
        and facts_equal
        and left.requirements == right.requirements
        and left.constraints == right.constraints
    )


def _fact_diagnostics(
    checkpoints: tuple[TrajectoryCheckpoint, ...],
) -> tuple[str, ...]:
    if not any(item.facts or item.fact_capture_complete for item in checkpoints):
        return ()
    diagnostics: list[str] = []
    for checkpoint in checkpoints:
        prefix = f"checkpoint:{checkpoint.checkpoint_id}"
        if not checkpoint.fact_capture_complete:
            diagnostics.append(f"{prefix}:Fact capture incomplete")
        if not checkpoint.facts:
            diagnostics.append(f"{prefix}:no explicit Environment Facts")
        for fact in checkpoint.facts:
            if fact.validity in (FactValidity.STALE, FactValidity.UNKNOWN):
                diagnostics.append(
                    f"{prefix}:Fact {fact.fact_id!r} validity is {fact.validity.value}"
                )
    return tuple(diagnostics)


class ShadowTrajectoryAnalyzer:
    """Normalize one RunAudit and assess caller-supplied environment checkpoints."""

    def __init__(self, *, max_period: int = 3) -> None:
        if isinstance(max_period, bool) or not isinstance(max_period, int):
            raise TypeError("max_period must be an integer")
        if max_period <= 0:
            raise ValueError("max_period must be greater than zero")
        self._max_period = max_period

    def analyze(
        self,
        audit: RunAudit,
        checkpoints: Iterable[TrajectoryCheckpoint],
    ) -> TrajectoryReport:
        if not isinstance(audit, RunAudit):
            raise TypeError("audit must be a RunAudit")
        assessment = self.assess(checkpoints)
        actions, observations, audit_diagnostics = _normalize_audit(audit.records)
        return self._report(
            audit,
            assessment,
            actions,
            observations,
            audit_diagnostics,
        )

    def assess(
        self,
        checkpoints: Iterable[TrajectoryCheckpoint],
    ) -> TrajectoryAssessment:
        """Assess Checkpoints before a Run has reached a terminal audit State."""

        checkpoint_values = tuple(checkpoints)
        if not all(
            isinstance(item, TrajectoryCheckpoint) for item in checkpoint_values
        ):
            raise TypeError("checkpoints must contain TrajectoryCheckpoint values")
        if not checkpoint_values:
            return TrajectoryAssessment(
                verdict=TrajectoryVerdict.INSUFFICIENT_EVIDENCE,
                period=None,
                candidate_checkpoint_ids=(),
                repeated_action_path=False,
                repeated_action_signatures=(),
                task_progress_over_repeated_window=None,
                progress=(),
                diagnostics=("no environment checkpoints",),
            )
        progress, checkpoint_diagnostics = _progress(checkpoint_values)
        fact_diagnostics = _fact_diagnostics(checkpoint_values[-1:])
        diagnostics = checkpoint_diagnostics + fact_diagnostics
        if checkpoint_diagnostics or fact_diagnostics:
            return TrajectoryAssessment(
                verdict=TrajectoryVerdict.INSUFFICIENT_EVIDENCE,
                period=None,
                candidate_checkpoint_ids=(),
                repeated_action_path=False,
                repeated_action_signatures=(),
                task_progress_over_repeated_window=None,
                progress=progress,
                diagnostics=diagnostics,
            )
        confirmed = self._confirmed_cycle(checkpoint_values, progress)
        if confirmed is not None:
            period, candidates, task_progress = confirmed
            return TrajectoryAssessment(
                verdict=TrajectoryVerdict.NON_PROGRESS_CYCLE,
                period=period,
                candidate_checkpoint_ids=tuple(
                    item.checkpoint_id for item in candidates
                ),
                repeated_action_path=True,
                repeated_action_signatures=tuple(
                    signature
                    for checkpoint in candidates[-period:]
                    for signature in checkpoint.causal_action_signatures
                ),
                task_progress_over_repeated_window=task_progress,
                progress=progress,
                diagnostics=diagnostics,
            )
        suspected = self._suspected_cycle(checkpoint_values, progress)
        if suspected is not None:
            period, candidates, task_progress = suspected
            return TrajectoryAssessment(
                verdict=TrajectoryVerdict.CYCLE_SUSPECTED,
                period=period,
                candidate_checkpoint_ids=tuple(
                    item.checkpoint_id for item in candidates
                ),
                repeated_action_path=False,
                repeated_action_signatures=(),
                task_progress_over_repeated_window=task_progress,
                progress=progress,
                diagnostics=diagnostics,
            )
        incomplete = self._ambiguous_cycle(checkpoint_values)
        if incomplete:
            return TrajectoryAssessment(
                verdict=TrajectoryVerdict.CAUSALLY_AMBIGUOUS,
                period=None,
                candidate_checkpoint_ids=tuple(
                    item.checkpoint_id for item in incomplete
                ),
                repeated_action_path=False,
                repeated_action_signatures=(),
                task_progress_over_repeated_window=None,
                progress=progress,
                diagnostics=diagnostics
                + tuple(
                    (
                        f"checkpoint:{item.checkpoint_id}:causal attribution incomplete: "
                        f"{item.causal_exclusion_reason}"
                    )
                    for item in incomplete
                ),
            )
        return TrajectoryAssessment(
            verdict=TrajectoryVerdict.NO_CYCLE,
            period=None,
            candidate_checkpoint_ids=(),
            repeated_action_path=False,
            repeated_action_signatures=(),
            task_progress_over_repeated_window=None,
            progress=progress,
            diagnostics=diagnostics,
        )

    def _confirmed_cycle(
        self,
        checkpoints: tuple[TrajectoryCheckpoint, ...],
        progress: tuple[ProgressSnapshot, ...],
    ) -> tuple[int, tuple[TrajectoryCheckpoint, ...], float | None] | None:
        transitions = [item.causal_action_signatures for item in checkpoints[1:]]
        for period in range(1, self._max_period + 1):
            window_size = period * 2 + 1
            if len(checkpoints) < window_size:
                continue
            candidates = checkpoints[-window_size:]
            if not all(
                _equivalent(candidates[offset], candidates[period + offset])
                for offset in range(period + 1)
            ):
                continue
            action_window = transitions[-period * 2 :]
            if (
                len(action_window) != period * 2
                or not all(action_window)
                or action_window[:period] != action_window[period:]
            ):
                continue
            repeated_progress = progress[-period:]
            requirement_progress = sum(
                item.requirement_coverage_delta for item in repeated_progress
            )
            task_deltas = tuple(item.task_progress_delta for item in repeated_progress)
            task_progress = (
                None
                if any(item is None for item in task_deltas)
                else sum(item for item in task_deltas if item is not None)
            )
            best_before_repeat = progress[-period - 1].best_requirement_coverage
            if (
                all(item.causally_complete for item in candidates)
                and not _fact_diagnostics(candidates)
                and requirement_progress <= 0
                and progress[-1].best_requirement_coverage <= best_before_repeat
                and all(not item.new_evidence for item in repeated_progress)
                and sum(item.actor_actions_since_previous for item in repeated_progress)
                > 0
            ):
                return period, candidates, task_progress
        return None

    def _suspected_cycle(
        self,
        checkpoints: tuple[TrajectoryCheckpoint, ...],
        progress: tuple[ProgressSnapshot, ...],
    ) -> tuple[int, tuple[TrajectoryCheckpoint, ...], float | None] | None:
        for period in range(1, self._max_period + 1):
            window_size = period * 2
            if len(checkpoints) < window_size:
                continue
            candidates = checkpoints[-window_size:]
            if not all(
                _equivalent(candidates[offset], candidates[period + offset])
                for offset in range(period)
            ):
                continue
            repeated_progress = progress[-period:]
            requirement_progress = sum(
                item.requirement_coverage_delta for item in repeated_progress
            )
            task_deltas = tuple(item.task_progress_delta for item in repeated_progress)
            task_progress = (
                None
                if any(item is None for item in task_deltas)
                else sum(item for item in task_deltas if item is not None)
            )
            if (
                all(item.causally_complete for item in candidates)
                and not _fact_diagnostics(candidates)
                and requirement_progress <= 0
                and all(not item.new_evidence for item in repeated_progress)
                and sum(item.actor_actions_since_previous for item in repeated_progress)
                > 0
            ):
                return period, candidates, task_progress
        return None

    def _ambiguous_cycle(
        self,
        checkpoints: tuple[TrajectoryCheckpoint, ...],
    ) -> tuple[TrajectoryCheckpoint, ...]:
        for period in range(1, self._max_period + 1):
            for window_size in (period * 2 + 1, period * 2):
                if len(checkpoints) < window_size:
                    continue
                candidates = checkpoints[-window_size:]
                comparisons = period + 1 if window_size % 2 else period
                if not all(
                    _equivalent(candidates[offset], candidates[period + offset])
                    for offset in range(comparisons)
                ):
                    continue
                incomplete = tuple(
                    item for item in candidates if not item.causally_complete
                )
                if incomplete:
                    return incomplete
        return ()

    @staticmethod
    def _report(
        audit: RunAudit,
        assessment: TrajectoryAssessment,
        actions: tuple[NormalizedAction, ...],
        observations: tuple[NormalizedObservation, ...],
        audit_diagnostics: tuple[str, ...],
    ) -> TrajectoryReport:
        return TrajectoryReport(
            run_id=audit.run_id,
            run_status=audit.result.status,
            stop_reason=audit.result.stop_reason,
            committed=audit.committed,
            base_revision=audit.base_revision,
            resulting_revision=audit.resulting_revision,
            turns=audit.result.turns,
            total_tokens=audit.result.usage.total_tokens,
            request_count=audit.result.usage.request_count,
            verdict=assessment.verdict,
            period=assessment.period,
            candidate_checkpoint_ids=assessment.candidate_checkpoint_ids,
            repeated_action_path=assessment.repeated_action_path,
            repeated_action_signatures=assessment.repeated_action_signatures,
            task_progress_over_repeated_window=(
                assessment.task_progress_over_repeated_window
            ),
            actions=actions,
            observations=observations,
            progress=assessment.progress,
            diagnostics=audit_diagnostics + assessment.diagnostics,
        )


CheckpointSource = Callable[[str], Iterable[TrajectoryCheckpoint]]
ReportSink = Callable[[TrajectoryReport], Awaitable[None]]


class ShadowTrajectoryObserver:
    """Join post-commit RunAudit with host-owned Facts and emit a detached report."""

    def __init__(
        self,
        *,
        checkpoint_source: CheckpointSource,
        report_sink: ReportSink,
        analyzer: ShadowTrajectoryAnalyzer | None = None,
    ) -> None:
        if not callable(checkpoint_source):
            raise TypeError("checkpoint_source must be callable")
        if not callable(report_sink):
            raise TypeError("report_sink must be callable")
        self._checkpoint_source = checkpoint_source
        self._report_sink = report_sink
        self._analyzer = analyzer or ShadowTrajectoryAnalyzer()

    async def observe(self, audit: RunAudit) -> None:
        checkpoints = self._checkpoint_source(audit.run_id)
        report = self._analyzer.analyze(audit, checkpoints)
        await self._report_sink(report)
