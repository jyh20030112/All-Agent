from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from simagentplg.agent.usage import RunUsage


class RunStatus(StrEnum):
    """Terminal status of one orchestrator run."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class StopReason(StrEnum):
    """Reason why an orchestrator run stopped."""

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
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Structured terminal result produced by ``AgentOrchestrator``."""

    status: RunStatus
    stop_reason: StopReason
    turns: int
    output: str | None = None
    error: str | None = None
    usage: RunUsage = field(default_factory=RunUsage)

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.COMPLETED

    def raise_for_status(self) -> None:
        """Raise a compatibility error when the run did not complete."""

        if not self.succeeded:
            raise AgentRunError(self)


class AgentRunError(RuntimeError):
    """Compatibility exception wrapping a structured failed run result."""

    def __init__(self, result: AgentRunResult) -> None:
        self.result = result
        super().__init__(result.error or result.stop_reason.value)
