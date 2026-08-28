from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from ejagent.contracts.json import JsonObject, freeze_json_object
from ejagent.contracts.messages import ConversationMessage, is_conversation_message
from ejagent.contracts.usage import RunUsage


class RunIntent(StrEnum):
    """Why a Harness started one Run."""

    TASK = "task"
    CONTINUE = "continue"


class RunStatus(StrEnum):
    """Terminal status of one Runtime Kernel Run."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class StopReason(StrEnum):
    """Stable reason why one Run stopped."""

    TEXT_RESPONSE = "text_response"
    TOOL_COMPLETION = "tool_completion"
    TOOL_REJECTED = "tool_rejected"
    TOOL_CANCELLED = "tool_cancelled"
    BEHAVIOR_STOP = "behavior_stop"
    EXTERNAL_ABORT = "external_abort"
    EMPTY_RESPONSE = "empty_response"
    MAX_STEPS = "max_steps"
    MAX_NO_TOOL_RESPONSES = "max_no_tool_responses"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    USAGE_UNAVAILABLE = "usage_unavailable"
    CONTEXT_OVERFLOW = "context_overflow"
    COMPACTION_FAILED = "compaction_failed"
    PERSISTENCE_FAILED = "persistence_failed"
    RUNTIME_ERROR = "runtime_error"


class RunPhase(StrEnum):
    """Execution phase associated with one structured Run failure."""

    PREPARATION = "preparation"
    CONTEXT = "context"
    MODEL = "model"
    TOOL = "tool"
    CONTROL = "control"
    COMMIT = "commit"
    RUNTIME = "runtime"


class FailureCode(StrEnum):
    """Provider-neutral category for an expected operational failure."""

    PROVIDER_ERROR = "provider_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    TOOL_ERROR = "tool_error"
    POLICY_REJECTED = "policy_rejected"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    CONTEXT_OVERFLOW = "context_overflow"
    COMPACTION_FAILED = "compaction_failed"
    PERSISTENCE_FAILED = "persistence_failed"
    RUNTIME_ERROR = "runtime_error"


def _positive_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class RunLimits:
    """Immutable limits captured when one Run starts."""

    max_turns: int = 20
    max_tokens: int | None = None
    max_repeated_tool_calls: int = 3

    def __post_init__(self) -> None:
        _positive_integer(self.max_turns, "max_turns")
        _positive_integer(
            self.max_repeated_tool_calls,
            "max_repeated_tool_calls",
        )
        if self.max_tokens is not None:
            _positive_integer(self.max_tokens, "max_tokens")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Immutable input captured by a Harness for one Kernel Run."""

    run_id: str
    base_revision: int
    intent: RunIntent
    task: str | None
    messages: tuple[ConversationMessage, ...]
    limits: RunLimits = field(default_factory=RunLimits)
    configuration_revision: str = "default"
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if isinstance(self.base_revision, bool) or not isinstance(
            self.base_revision,
            int,
        ):
            raise TypeError("base_revision must be an integer")
        if self.base_revision < 0:
            raise ValueError("base_revision must not be negative")
        if not isinstance(self.intent, RunIntent):
            raise TypeError("intent must be a RunIntent")
        if self.intent is RunIntent.TASK:
            if not isinstance(self.task, str) or not self.task.strip():
                raise ValueError("task Run requires a non-empty task")
        elif self.task is not None:
            raise ValueError("Continue Run must not contain a task")
        object.__setattr__(self, "messages", tuple(self.messages))
        if not all(is_conversation_message(message) for message in self.messages):
            raise TypeError("messages must contain ConversationMessage values")
        if not isinstance(self.limits, RunLimits):
            raise TypeError("limits must be RunLimits")
        if (
            not isinstance(self.configuration_revision, str)
            or not self.configuration_revision.strip()
        ):
            raise ValueError("configuration_revision must not be empty")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a JSON object")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object(self.metadata, label="RunSpec metadata"),
        )


@dataclass(frozen=True, slots=True)
class RunDelta:
    """Conversation messages proposed against one Harness revision."""

    base_revision: int
    messages: tuple[ConversationMessage, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.base_revision, bool) or not isinstance(
            self.base_revision,
            int,
        ):
            raise TypeError("base_revision must be an integer")
        if self.base_revision < 0:
            raise ValueError("base_revision must not be negative")
        object.__setattr__(self, "messages", tuple(self.messages))
        if not all(is_conversation_message(message) for message in self.messages):
            raise TypeError("messages must contain ConversationMessage values")

    @property
    def next_revision(self) -> int:
        """Return the revision produced when this Delta is committed."""

        return self.base_revision + 1


@dataclass(frozen=True, slots=True)
class RunFailure:
    """Structured expected failure produced during one Run."""

    phase: RunPhase
    code: FailureCode
    message: str
    retryable: bool = False
    cause: BaseException | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.phase, RunPhase):
            raise TypeError("failure phase must be a RunPhase")
        if not isinstance(self.code, FailureCode):
            raise TypeError("failure code must be a FailureCode")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("failure message must not be empty")
        if not isinstance(self.retryable, bool):
            raise TypeError("failure retryable must be a boolean")
        if self.cause is not None and not isinstance(self.cause, BaseException):
            raise TypeError("failure cause must be an exception or None")


@dataclass(frozen=True, slots=True)
class RunResult:
    """Provider-neutral terminal summary of one Kernel Run."""

    run_id: str
    status: RunStatus
    stop_reason: StopReason
    turns: int
    output: str | None = None
    usage: RunUsage = field(default_factory=RunUsage)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("result run_id must not be empty")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if not isinstance(self.stop_reason, StopReason):
            raise TypeError("stop_reason must be a StopReason")
        if isinstance(self.turns, bool) or not isinstance(self.turns, int):
            raise TypeError("turns must be an integer")
        if self.turns < 0:
            raise ValueError("turns must not be negative")
        if self.output is not None and not isinstance(self.output, str):
            raise TypeError("output must be a string or None")
        if not isinstance(self.usage, RunUsage):
            raise TypeError("usage must be RunUsage")

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One immutable fact emitted while a Run was executing."""

    run_id: str
    sequence: int
    kind: str
    occurred_at: datetime
    payload: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("audit run_id must not be empty")
        _positive_integer(self.sequence, "audit sequence")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("audit kind must not be empty")
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("audit occurred_at must be a datetime")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("audit occurred_at must be timezone-aware")
        if not isinstance(self.payload, Mapping):
            raise TypeError("audit payload must be a JSON object")
        object.__setattr__(
            self,
            "payload",
            freeze_json_object(self.payload, label="audit payload"),
        )


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Complete Kernel output awaiting a Harness commit decision."""

    result: RunResult
    delta: RunDelta
    audit_records: tuple[AuditRecord, ...] = ()
    failure: RunFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, RunResult):
            raise TypeError("result must be a RunResult")
        if not isinstance(self.delta, RunDelta):
            raise TypeError("delta must be a RunDelta")
        object.__setattr__(self, "audit_records", tuple(self.audit_records))
        if not all(isinstance(record, AuditRecord) for record in self.audit_records):
            raise TypeError("audit_records must contain AuditRecord values")
        if any(record.run_id != self.result.run_id for record in self.audit_records):
            raise ValueError("audit records must belong to the outcome Run")
        sequences = [record.sequence for record in self.audit_records]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("audit record sequences must be unique and ordered")
        if self.result.status is RunStatus.FAILED:
            if self.failure is None:
                raise ValueError("failed RunOutcome requires a RunFailure")
        elif self.failure is not None:
            raise ValueError("non-failed RunOutcome must not contain a RunFailure")
