from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ejagent.contracts.audit import RunAudit
from ejagent.contracts.conversation import ConversationSnapshot
from ejagent.contracts.messages import ConversationMessage
from ejagent.contracts.runs import RunOutcome, RunResult, RunStatus


def _required_id(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Harness recovery state with Conversation kept as its own data domain."""

    agent_id: str
    conversation: ConversationSnapshot = field(default_factory=ConversationSnapshot)
    last_result: RunResult | None = None

    def __post_init__(self) -> None:
        _required_id(self.agent_id, "snapshot agent_id")
        if not isinstance(self.conversation, ConversationSnapshot):
            raise TypeError("snapshot conversation must be a ConversationSnapshot")
        if self.last_result is not None and not isinstance(self.last_result, RunResult):
            raise TypeError("snapshot last_result must be a RunResult or None")

    @property
    def revision(self) -> int:
        return self.conversation.revision

    @property
    def messages(self) -> tuple[ConversationMessage, ...]:
        return self.conversation.messages


@dataclass(frozen=True, slots=True)
class SessionCommit:
    """One idempotent Run record proposed against a committed revision."""

    agent_id: str
    base: ConversationSnapshot
    outcome: RunOutcome

    def __post_init__(self) -> None:
        _required_id(self.agent_id, "commit agent_id")
        if not isinstance(self.base, ConversationSnapshot):
            raise TypeError("commit base must be a ConversationSnapshot")
        if not isinstance(self.outcome, RunOutcome):
            raise TypeError("commit outcome must be a RunOutcome")
        if self.outcome.delta.base_revision != self.base_revision:
            raise ValueError("commit and Delta base revisions must match")

    @property
    def base_revision(self) -> int:
        return self.base.revision

    @property
    def base_messages(self) -> tuple[ConversationMessage, ...]:
        return self.base.messages

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
    def resulting_conversation(self) -> ConversationSnapshot:
        if not self.advances_revision:
            return self.base
        return ConversationSnapshot(
            revision=self.resulting_revision,
            messages=(*self.base.messages, *self.outcome.delta.messages),
        )

    @property
    def audit(self) -> RunAudit:
        return RunAudit(
            result=self.outcome.result,
            base_revision=self.base_revision,
            resulting_revision=self.resulting_revision,
            committed=self.advances_revision,
            records=self.outcome.audit_records,
            failure=self.outcome.failure,
        )


class SessionStoreError(RuntimeError):
    """Expected operational failure at the durable Session seam."""


class SessionConflictError(SessionStoreError):
    """A Session commit was based on a stale revision or Conversation."""


class SessionStoreSerializationError(SessionStoreError):
    """Durable Session data is malformed, unsupported, or not JSON-compatible."""


class SessionStoreLockTimeoutError(SessionStoreError):
    """A durable Session lock could not be acquired within its timeout."""


class SessionMigrationError(SessionStoreError):
    """Legacy Session data cannot be converted without losing semantics."""

    def __init__(
        self,
        message: str,
        *,
        remediation: str,
    ) -> None:
        if not message.strip():
            raise ValueError("migration error message must not be empty")
        if not remediation.strip():
            raise ValueError("migration remediation must not be empty")
        self.remediation = remediation
        super().__init__(f"{message} Remediation: {remediation}")


class SessionStore(Protocol):
    """Durable compare-and-commit seam owned by an AgentHarness."""

    async def load(self, agent_id: str) -> SessionSnapshot | None:
        """Load the latest committed snapshot, if one exists."""

    async def commit(self, commit: SessionCommit) -> SessionSnapshot:
        """Atomically record one Run and return its resulting snapshot.

        Repeating the same commit for the same ``run_id`` must be idempotent.
        A different commit using an existing ``run_id`` must fail.
        """
