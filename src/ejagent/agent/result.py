from __future__ import annotations

from dataclasses import dataclass, field

from ejagent.contracts.runs import RunStatus, StopReason
from ejagent.contracts.usage import RunUsage


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
