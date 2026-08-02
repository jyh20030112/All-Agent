from __future__ import annotations

import asyncio

from ejagent.contracts.audit import AuditReader, RunAudit
from ejagent.contracts.session import (
    SessionCommit,
    SessionConflictError,
    SessionSnapshot,
    SessionStore,
)


class MemorySessionStore(SessionStore, AuditReader):
    """Process-local atomic SessionStore with idempotent Run commits."""

    def __init__(self) -> None:
        self._snapshots: dict[str, SessionSnapshot] = {}
        self._commits: dict[tuple[str, str], tuple[SessionCommit, SessionSnapshot]] = {}
        self._order: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def load(self, agent_id: str) -> SessionSnapshot | None:
        async with self._lock:
            return self._snapshots.get(agent_id)

    async def commit(self, commit: SessionCommit) -> SessionSnapshot:
        async with self._lock:
            key = (commit.agent_id, commit.run_id)
            existing = self._commits.get(key)
            if existing is not None:
                previous, snapshot = existing
                if previous != commit:
                    raise SessionConflictError(
                        f"run_id {commit.run_id!r} already identifies "
                        "a different commit"
                    )
                return snapshot

            current = self._snapshots.get(commit.agent_id)
            if current is None:
                if commit.base_revision != 0:
                    raise SessionConflictError(
                        f"agent {commit.agent_id!r} has no revision "
                        f"{commit.base_revision}"
                    )
                current = SessionSnapshot(
                    agent_id=commit.agent_id,
                    conversation=commit.base,
                )

            if current.revision != commit.base_revision:
                raise SessionConflictError(
                    f"agent {commit.agent_id!r} is at revision "
                    f"{current.revision}, not {commit.base_revision}"
                )
            if current.messages != commit.base_messages:
                raise SessionConflictError(
                    f"agent {commit.agent_id!r} Conversation does not match "
                    f"revision {commit.base_revision}"
                )

            snapshot = SessionSnapshot(
                agent_id=commit.agent_id,
                conversation=commit.resulting_conversation,
                last_result=(
                    commit.outcome.result
                    if commit.advances_revision
                    else current.last_result
                ),
            )
            self._snapshots[commit.agent_id] = snapshot
            self._commits[key] = (commit, snapshot)
            self._order.setdefault(commit.agent_id, []).append(commit.run_id)
            return snapshot

    async def load_audit(self, agent_id: str) -> tuple[RunAudit, ...]:
        """Return Run audit facts without exposing Conversation deltas."""

        async with self._lock:
            return tuple(
                self._commits[(agent_id, run_id)][0].audit
                for run_id in self._order.get(agent_id, ())
            )

    async def commits(self, agent_id: str) -> tuple[SessionCommit, ...]:
        """Return recorded commits in insertion order for inspection."""

        async with self._lock:
            return tuple(
                self._commits[(agent_id, run_id)][0]
                for run_id in self._order.get(agent_id, ())
            )
