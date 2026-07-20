from typing import Protocol, runtime_checkable

from simagentplg.session.journal import (
    DEFAULT_SESSION_BRANCH,
    SessionRecord,
    SessionRecordDraft,
)
from simagentplg.session.tree import SessionBranch, SessionCheckout, SessionRetry
from simagentplg.session.types import AgentSession


class SessionStorage(Protocol):
    """Persistence boundary for detached Agent Session snapshots."""

    async def load(self, session_id: str) -> AgentSession | None:
        """Load a detached Session or return ``None`` when it does not exist."""

    async def save(self, session: AgentSession) -> None:
        """Create or replace one Session snapshot."""


@runtime_checkable
class SessionJournalStorage(SessionStorage, Protocol):
    """Storage capable of appending semantic Session journal records."""

    async def checkout(
        self,
        session_id: str,
        *,
        branch_id: str = DEFAULT_SESSION_BRANCH,
        record_id: str | None = None,
    ) -> SessionCheckout | None:
        """Project one branch head or an exact record."""

    async def append(
        self,
        draft: SessionRecordDraft,
        *,
        expected_head_id: str | None = None,
        check_head: bool = False,
    ) -> SessionRecord:
        """Atomically append one mutation and return its assigned envelope."""


@runtime_checkable
class SessionTreeStorage(SessionJournalStorage, Protocol):
    """Backend-neutral persistence boundary for addressable Session trees."""

    async def head(
        self,
        session_id: str,
        *,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecord | None:
        """Return the immutable record currently heading one branch."""

    async def list_branches(self, session_id: str) -> tuple[SessionBranch, ...]:
        """Return the Session's branches in backend-defined stable order."""

    async def records(self, session_id: str) -> tuple[SessionRecord, ...]:
        """Return immutable records in global revision order."""

    async def fork(
        self,
        session_id: str,
        *,
        source_branch: str = DEFAULT_SESSION_BRANCH,
        from_record_id: str | None = None,
        branch_id: str | None = None,
    ) -> SessionCheckout:
        """Create a branch at a completed source projection."""

    async def rollback(
        self,
        session_id: str,
        *,
        to_record_id: str,
        source_branch: str = DEFAULT_SESSION_BRANCH,
        branch_id: str | None = None,
    ) -> SessionCheckout:
        """Create a branch at an ancestor without rewriting its source."""

    async def prepare_retry(
        self,
        session_id: str,
        *,
        run_id: str,
        source_branch: str = DEFAULT_SESSION_BRANCH,
        branch_id: str | None = None,
    ) -> SessionRetry:
        """Branch before one Run and return its original task."""
