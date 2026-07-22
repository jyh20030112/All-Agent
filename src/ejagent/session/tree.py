from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ejagent.session.journal import SessionRecord
from ejagent.session.types import AgentSession


class SessionBranchIntent(StrEnum):
    """Audited reason for creating a Session branch."""

    FORK = "fork"
    ROLLBACK = "rollback"
    RETRY = "retry"


@dataclass(frozen=True, slots=True)
class SessionBranch:
    """One named branch and its current immutable journal head."""

    branch_id: str
    head_record_id: str
    base_record_id: str | None
    source_branch_id: str | None
    intent: SessionBranchIntent | None
    created_revision: int

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("branch_id must not be empty")
        if not self.head_record_id:
            raise ValueError("head_record_id must not be empty")
        if self.created_revision <= 0:
            raise ValueError("created_revision must be greater than zero")


@dataclass(frozen=True, slots=True)
class SessionCheckout:
    """Detached Session projection at one branch or record head."""

    session: AgentSession
    branch: SessionBranch
    head: SessionRecord

    def __post_init__(self) -> None:
        object.__setattr__(self, "session", self.session.snapshot())


@dataclass(frozen=True, slots=True)
class SessionRetry:
    """A prepared retry branch and the original Run task to execute."""

    checkout: SessionCheckout
    task: str
    retried_run_id: str

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must not be empty")
        if not self.retried_run_id:
            raise ValueError("retried_run_id must not be empty")
