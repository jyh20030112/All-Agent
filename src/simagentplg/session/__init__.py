"""Durable Agent Session projections built from lifecycle event trees."""

from simagentplg.session.codec import (
    SESSION_SCHEMA_VERSION,
    session_from_dict,
    session_to_dict,
)
from simagentplg.session.errors import (
    SessionConflictError,
    SessionError,
    SessionLockTimeoutError,
    SessionSerializationError,
    SessionStorageError,
)
from simagentplg.session.journal import (
    DEFAULT_SESSION_BRANCH,
    SESSION_JOURNAL_SCHEMA_VERSION,
    SessionRecord,
    SessionRecordDraft,
    SessionRecordKind,
)
from simagentplg.session.jsonl import JsonlSessionStorage
from simagentplg.session.memory import MemorySessionStorage
from simagentplg.session.recorder import SessionRecorder
from simagentplg.session.storage import (
    SessionJournalStorage,
    SessionStorage,
    SessionTreeStorage,
)
from simagentplg.session.tree import (
    SessionBranch,
    SessionBranchIntent,
    SessionCheckout,
    SessionRetry,
)
from simagentplg.session.types import (
    AgentSession,
    SessionCompaction,
    SessionMessage,
    SessionRun,
)

__all__ = [
    "AgentSession",
    "SessionMessage",
    "SessionRun",
    "SessionCompaction",
    "SessionStorage",
    "SessionJournalStorage",
    "SessionTreeStorage",
    "JsonlSessionStorage",
    "MemorySessionStorage",
    "SessionRecorder",
    "SESSION_SCHEMA_VERSION",
    "session_to_dict",
    "session_from_dict",
    "SessionError",
    "SessionConflictError",
    "SessionLockTimeoutError",
    "SessionSerializationError",
    "SessionStorageError",
    "SESSION_JOURNAL_SCHEMA_VERSION",
    "DEFAULT_SESSION_BRANCH",
    "SessionRecordKind",
    "SessionRecordDraft",
    "SessionRecord",
    "SessionBranchIntent",
    "SessionBranch",
    "SessionCheckout",
    "SessionRetry",
]
