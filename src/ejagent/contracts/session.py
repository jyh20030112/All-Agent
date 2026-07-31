from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ejagent.contracts.messages import ConversationMessage, is_conversation_message
from ejagent.contracts.runs import RunOutcome, RunResult, RunStatus


def _required_id(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _revision(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Immutable committed Conversation state loaded by an AgentHarness."""

    agent_id: str
    revision: int = 0
    messages: tuple[ConversationMessage, ...] = ()
    last_result: RunResult | None = None

    def __post_init__(self) -> None:
        _required_id(self.agent_id, "snapshot agent_id")
        _revision(self.revision, "snapshot revision")
        object.__setattr__(self, "messages", tuple(self.messages))
        if not all(is_conversation_message(message) for message in self.messages):
            raise TypeError("snapshot messages must contain ConversationMessage values")
        if self.last_result is not None and not isinstance(self.last_result, RunResult):
            raise TypeError("snapshot last_result must be a RunResult or None")


@dataclass(frozen=True, slots=True)
class SessionCommit:
    """One idempotent Run record proposed against a committed revision."""

    agent_id: str
    base_revision: int
    base_messages: tuple[ConversationMessage, ...]
    outcome: RunOutcome

    def __post_init__(self) -> None:
        _required_id(self.agent_id, "commit agent_id")
        _revision(self.base_revision, "commit base_revision")
        object.__setattr__(self, "base_messages", tuple(self.base_messages))
        if not all(is_conversation_message(message) for message in self.base_messages):
            raise TypeError(
                "commit base_messages must contain ConversationMessage values"
            )
        if not isinstance(self.outcome, RunOutcome):
            raise TypeError("commit outcome must be a RunOutcome")
        if self.outcome.delta.base_revision != self.base_revision:
            raise ValueError("commit and Delta base revisions must match")

    @property
    def run_id(self) -> str:
        return self.outcome.result.run_id

    @property
    def advances_revision(self) -> bool:
        """Return whether this Run is eligible for Conversation commit."""

        return self.outcome.result.status is RunStatus.COMPLETED

    @property
    def resulting_revision(self) -> int:
        return self.base_revision + int(self.advances_revision)

    @property
    def resulting_messages(self) -> tuple[ConversationMessage, ...]:
        if not self.advances_revision:
            return self.base_messages
        return (*self.base_messages, *self.outcome.delta.messages)


class SessionStoreError(RuntimeError):
    """Expected operational failure at the durable Session seam."""


class SessionConflictError(SessionStoreError):
    """A Session commit was based on a stale revision or Conversation."""


class SessionStore(Protocol):
    """Durable compare-and-commit seam owned by an AgentHarness."""

    async def load(self, agent_id: str) -> SessionSnapshot | None:
        """Load the latest committed snapshot, if one exists."""

    async def commit(self, commit: SessionCommit) -> SessionSnapshot:
        """Atomically record one Run and return its resulting snapshot.

        Repeating the same commit for the same ``run_id`` must be idempotent.
        A different commit using an existing ``run_id`` must fail.
        """
