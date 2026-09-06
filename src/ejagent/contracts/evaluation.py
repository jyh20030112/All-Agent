"""Immutable acceptance criteria and bounded execution observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CompletionMode(StrEnum):
    OBSERVE = "observe"
    ENFORCE = "enforce"


@dataclass(frozen=True, slots=True)
class CompletionPolicy:
    """Independent opt-in enforcement, bounded by both retry and Run limits."""

    mode: CompletionMode = CompletionMode.OBSERVE
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CompletionMode):
            raise TypeError("completion mode must be CompletionMode")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("completion max_retries must be a non-negative integer")


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


@dataclass(frozen=True, slots=True)
class EvaluationCriterion:
    """One host-declared condition and its deterministic verification method."""

    criterion_id: str
    description: str
    method: str
    evidence_keys: tuple[str, ...]
    semantic: bool = False
    guard_method: str | None = None
    completion_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.semantic, bool):
            raise TypeError("semantic must be boolean")
        if not isinstance(self.completion_only, bool):
            raise TypeError("completion_only must be boolean")
        if self.guard_method is not None:
            _text(self.guard_method, "guard_method")
            if not self.semantic:
                raise ValueError("guard_method is only used for semantic criteria")
        for name in ("criterion_id", "description", "method"):
            _text(getattr(self, name), name)
        keys = tuple(self.evidence_keys)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("evidence_keys must be non-empty and unique")
        for key in keys:
            _text(key, "evidence key")
        object.__setattr__(self, "evidence_keys", keys)


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Fixed, ordered criteria for one Run; follow-ups bind their own plan."""

    goal: str
    version: str
    requirements: tuple[EvaluationCriterion, ...]
    constraints: tuple[EvaluationCriterion, ...] = ()
    artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.goal, "goal")
        _text(self.version, "version")
        for name in ("requirements", "constraints", "artifact_refs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.requirements:
            raise ValueError("an evaluation plan requires at least one requirement")
        items = (*self.requirements, *self.constraints)
        if not all(isinstance(item, EvaluationCriterion) for item in items):
            raise TypeError("plan items must be EvaluationCriterion values")
        ids = [item.criterion_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique across the plan")
        for ref in self.artifact_refs:
            _text(ref, "artifact reference")


@dataclass(frozen=True, slots=True)
class CompletionCandidate:
    """Bounded proposed final text; truncation must never imply full validation."""

    text: str
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or len(self.text) > 65_536:
            raise ValueError(
                "candidate text must be a string of at most 65536 characters"
            )
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be boolean")


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """A receipt reference for one tool in the current completed batch."""

    call_id: str
    tool_name: str
    evidence_ref: str
    is_error: bool

    def __post_init__(self) -> None:
        for name in ("call_id", "tool_name", "evidence_ref"):
            _text(getattr(self, name), name)
        if not isinstance(self.is_error, bool):
            raise TypeError("is_error must be boolean")
