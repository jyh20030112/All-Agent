from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from simagentplg.agent.compaction import CompactionResult, SummaryEntry
from simagentplg.agent.result import AgentRunResult
from simagentplg.agent.types import AgentMessage
from simagentplg.session.codec import (
    agent_run_result_from_dict,
    agent_run_result_to_dict,
    session_from_dict,
    session_to_dict,
)
from simagentplg.session.errors import SessionSerializationError
from simagentplg.session.types import AgentSession

SESSION_JOURNAL_SCHEMA_VERSION = 1
DEFAULT_SESSION_BRANCH = "main"


class SessionRecordKind(StrEnum):
    """Mutation represented by one immutable Session journal record."""

    CHECKPOINT = "checkpoint"
    BRANCH_CREATED = "branch_created"
    RUN_STARTED = "run_started"
    RUN_CONTINUED = "run_continued"
    MESSAGE_APPENDED = "message_appended"
    MESSAGES_APPENDED = "messages_appended"
    STEERING_APPLIED = "steering_applied"
    COMPACTION_APPLIED = "compaction_applied"
    RUN_FINISHED = "run_finished"


@dataclass(frozen=True, slots=True)
class SessionRecordDraft:
    """Semantic Session mutation before journal identity is assigned."""

    session_id: str
    agent_id: str | None
    sequence: int
    kind: SessionRecordKind
    data: dict[str, Any]
    branch_id: str = DEFAULT_SESSION_BRANCH

    def __post_init__(self) -> None:
        _validate_envelope_fields(
            session_id=self.session_id,
            agent_id=self.agent_id,
            sequence=self.sequence,
            branch_id=self.branch_id,
        )
        object.__setattr__(self, "data", deepcopy(self.data))

    @classmethod
    def checkpoint(cls, session: AgentSession) -> SessionRecordDraft:
        return cls(
            session_id=session.session_id,
            agent_id=session.agent_id,
            sequence=0,
            kind=SessionRecordKind.CHECKPOINT,
            data={"document": session_to_dict(session)},
        )

    @classmethod
    def branch_created(
        cls,
        *,
        session_id: str,
        agent_id: str | None,
        branch_id: str,
        base_record_id: str | None,
        source_branch_id: str,
        source_head_id: str,
        intent: str,
        retried_run_id: str | None = None,
    ) -> SessionRecordDraft:
        data: dict[str, Any] = {
            "base_record_id": base_record_id,
            "source_branch_id": source_branch_id,
            "source_head_id": source_head_id,
            "intent": intent,
        }
        if retried_run_id is not None:
            data["retried_run_id"] = retried_run_id
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            sequence=0,
            kind=SessionRecordKind.BRANCH_CREATED,
            data=data,
            branch_id=branch_id,
        )

    @classmethod
    def run_started(
        cls,
        *,
        session_id: str,
        agent_id: str,
        sequence: int,
        run_id: str,
        task: str,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecordDraft:
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            sequence=sequence,
            kind=SessionRecordKind.RUN_STARTED,
            data={"run_id": run_id, "task": task},
            branch_id=branch_id,
        )

    @classmethod
    def run_continued(
        cls,
        *,
        session_id: str,
        agent_id: str,
        sequence: int,
        run_id: str,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecordDraft:
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            sequence=sequence,
            kind=SessionRecordKind.RUN_CONTINUED,
            data={"run_id": run_id},
            branch_id=branch_id,
        )

    @classmethod
    def message_appended(
        cls,
        *,
        session_id: str,
        agent_id: str,
        sequence: int,
        run_id: str,
        message: AgentMessage,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecordDraft:
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            sequence=sequence,
            kind=SessionRecordKind.MESSAGE_APPENDED,
            data={"run_id": run_id, "message": deepcopy(message)},
            branch_id=branch_id,
        )

    @classmethod
    def messages_appended(
        cls,
        *,
        session_id: str,
        agent_id: str,
        sequence: int,
        run_id: str,
        messages: tuple[AgentMessage, ...],
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecordDraft:
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            sequence=sequence,
            kind=SessionRecordKind.MESSAGES_APPENDED,
            data={"run_id": run_id, "messages": deepcopy(list(messages))},
            branch_id=branch_id,
        )

    @classmethod
    def steering_applied(
        cls,
        *,
        session_id: str,
        agent_id: str,
        sequence: int,
        run_id: str,
        input_id: str,
        content: str,
        target_turn: int,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecordDraft:
        if target_turn <= 0:
            raise ValueError("target_turn must be greater than zero")
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            sequence=sequence,
            kind=SessionRecordKind.STEERING_APPLIED,
            data={
                "run_id": run_id,
                "input_id": input_id,
                "content": content,
                "target_turn": target_turn,
            },
            branch_id=branch_id,
        )

    @classmethod
    def compaction_applied(
        cls,
        *,
        session_id: str,
        agent_id: str,
        sequence: int,
        result: CompactionResult,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecordDraft:
        if result.summary is None:
            raise ValueError("completed compaction requires a SummaryEntry")
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            sequence=sequence,
            kind=SessionRecordKind.COMPACTION_APPLIED,
            data={
                "operation_id": result.operation_id,
                "summary": result.summary.to_dict(),
                "messages": deepcopy(list(result.messages)),
            },
            branch_id=branch_id,
        )

    @classmethod
    def run_finished(
        cls,
        *,
        session_id: str,
        agent_id: str,
        sequence: int,
        run_id: str,
        result: AgentRunResult,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecordDraft:
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            sequence=sequence,
            kind=SessionRecordKind.RUN_FINISHED,
            data={
                "run_id": run_id,
                "result": agent_run_result_to_dict(result),
            },
            branch_id=branch_id,
        )


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One immutable, tree-addressable JSONL Session journal record."""

    record_id: str
    parent_id: str | None
    branch_id: str
    revision: int
    session_id: str
    agent_id: str | None
    sequence: int
    kind: SessionRecordKind
    data: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must not be empty")
        if self.parent_id is not None and not self.parent_id:
            raise ValueError("parent_id must not be empty")
        if self.revision <= 0:
            raise ValueError("revision must be greater than zero")
        _validate_envelope_fields(
            session_id=self.session_id,
            agent_id=self.agent_id,
            sequence=self.sequence,
            branch_id=self.branch_id,
        )
        object.__setattr__(self, "data", deepcopy(self.data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "journal_schema_version": SESSION_JOURNAL_SCHEMA_VERSION,
            "record_id": self.record_id,
            "parent_id": self.parent_id,
            "branch_id": self.branch_id,
            "revision": self.revision,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "sequence": self.sequence,
            "type": self.kind.value,
            "data": deepcopy(self.data),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SessionRecord:
        try:
            version = _integer(
                value.get("journal_schema_version"),
                "journal_schema_version",
            )
            if version != SESSION_JOURNAL_SCHEMA_VERSION:
                raise SessionSerializationError(
                    f"unsupported journal_schema_version {version}; "
                    f"expected {SESSION_JOURNAL_SCHEMA_VERSION}"
                )
            raw_parent = value.get("parent_id")
            parent_id = None if raw_parent is None else _string(raw_parent, "parent_id")
            raw_agent = value.get("agent_id")
            agent_id = None if raw_agent is None else _string(raw_agent, "agent_id")
            data = value.get("data")
            if not isinstance(data, Mapping):
                raise SessionSerializationError("data must be an object")
            return cls(
                record_id=_string(value.get("record_id"), "record_id"),
                parent_id=parent_id,
                branch_id=_string(value.get("branch_id"), "branch_id"),
                revision=_integer(value.get("revision"), "revision"),
                session_id=_string(value.get("session_id"), "session_id"),
                agent_id=agent_id,
                sequence=_integer(value.get("sequence"), "sequence"),
                kind=SessionRecordKind(_string(value.get("type"), "type")),
                data=deepcopy(dict(data)),
            )
        except SessionSerializationError:
            raise
        except (TypeError, ValueError) as exc:
            raise SessionSerializationError(
                f"invalid Session journal record: {exc}"
            ) from exc


def apply_session_record(
    session: AgentSession | None,
    record: SessionRecord | SessionRecordDraft,
) -> AgentSession:
    """Apply one validated mutation to a detached Session projection."""

    if record.kind is SessionRecordKind.CHECKPOINT:
        document = record.data.get("document")
        if not isinstance(document, Mapping):
            raise SessionSerializationError(
                "checkpoint data.document must be an object"
            )
        restored = session_from_dict(document)
        if restored.session_id != record.session_id:
            raise SessionSerializationError(
                "checkpoint Session id does not match its journal envelope"
            )
        if restored.agent_id != record.agent_id:
            raise SessionSerializationError(
                "checkpoint Agent id does not match its journal envelope"
            )
        return restored

    if record.kind is SessionRecordKind.BRANCH_CREATED:
        active = session or AgentSession(session_id=record.session_id)
        if active.session_id != record.session_id:
            raise SessionSerializationError("journal contains multiple Session ids")
        if record.agent_id is not None:
            active.bind_agent(record.agent_id)
        return active

    active = session or AgentSession(session_id=record.session_id)
    if active.session_id != record.session_id:
        raise SessionSerializationError("journal contains multiple Session ids")
    if record.agent_id is None:
        raise SessionSerializationError(f"{record.kind.value} record requires agent_id")
    active.bind_agent(record.agent_id)

    if record.kind is SessionRecordKind.RUN_STARTED:
        active.begin_run(
            _data_string(record, "run_id"),
            _data_string(record, "task"),
            record.sequence,
        )
    elif record.kind is SessionRecordKind.RUN_CONTINUED:
        active.begin_continue(
            _data_string(record, "run_id"),
            record.sequence,
        )
    elif record.kind is SessionRecordKind.MESSAGE_APPENDED:
        active.append_message(
            _data_string(record, "run_id"),
            record.sequence,
            _data_message(record, "message"),
        )
    elif record.kind is SessionRecordKind.MESSAGES_APPENDED:
        messages = record.data.get("messages")
        if not isinstance(messages, list):
            raise SessionSerializationError(
                "messages_appended data.messages must be an array"
            )
        run_id = _data_string(record, "run_id")
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise SessionSerializationError(
                    f"messages_appended data.messages[{index}] must be an object"
                )
            active.append_message(
                run_id,
                record.sequence,
                deepcopy(dict(message)),
            )
    elif record.kind is SessionRecordKind.STEERING_APPLIED:
        _data_string(record, "input_id")
        target_turn = _integer(record.data.get("target_turn"), "target_turn")
        if target_turn <= 0:
            raise SessionSerializationError(
                "steering_applied data.target_turn must be greater than zero"
            )
        active.append_message(
            _data_string(record, "run_id"),
            record.sequence,
            {
                "role": "user",
                "content": _data_string(record, "content"),
            },
        )
    elif record.kind is SessionRecordKind.COMPACTION_APPLIED:
        summary = record.data.get("summary")
        messages = record.data.get("messages")
        if not isinstance(summary, Mapping):
            raise SessionSerializationError(
                "compaction_applied data.summary must be an object"
            )
        if not isinstance(messages, list):
            raise SessionSerializationError(
                "compaction_applied data.messages must be an array"
            )
        if any(not isinstance(message, Mapping) for message in messages):
            raise SessionSerializationError(
                "compaction_applied messages must contain only objects"
            )
        normalized_messages = tuple(deepcopy(dict(message)) for message in messages)
        active.apply_compaction(
            _data_string(record, "operation_id"),
            record.sequence,
            SummaryEntry.from_dict(summary),
            normalized_messages,
        )
    elif record.kind is SessionRecordKind.RUN_FINISHED:
        active.finish_run(
            _data_string(record, "run_id"),
            record.sequence,
            agent_run_result_from_dict(record.data.get("result")),
        )
    else:
        raise SessionSerializationError(
            f"unsupported Session record type {record.kind.value!r}"
        )
    return active


def _validate_envelope_fields(
    *,
    session_id: str,
    agent_id: str | None,
    sequence: int,
    branch_id: str,
) -> None:
    if not session_id.strip():
        raise ValueError("session_id must not be empty")
    if agent_id is not None and not agent_id.strip():
        raise ValueError("agent_id must not be empty")
    if sequence < 0:
        raise ValueError("sequence must not be negative")
    if not branch_id.strip():
        raise ValueError("branch_id must not be empty")


def _data_string(
    record: SessionRecord | SessionRecordDraft,
    key: str,
) -> str:
    return _string(record.data.get(key), f"{record.kind.value} data.{key}")


def _data_message(
    record: SessionRecord | SessionRecordDraft,
    key: str,
) -> AgentMessage:
    value = record.data.get(key)
    if not isinstance(value, Mapping):
        raise SessionSerializationError(
            f"{record.kind.value} data.{key} must be an object"
        )
    return deepcopy(dict(value))


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SessionSerializationError(f"{label} must be a string")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SessionSerializationError(f"{label} must be an integer")
    return value
