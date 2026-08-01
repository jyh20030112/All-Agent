from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ejagent.contracts.audit import RunAudit
from ejagent.contracts.conversation import ConversationSnapshot
from ejagent.contracts.json import thaw_json_value
from ejagent.contracts.messages import (
    AssistantMessage,
    ConversationMessage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ejagent.contracts.runs import (
    AuditRecord,
    FailureCode,
    RunDelta,
    RunFailure,
    RunOutcome,
    RunPhase,
    RunResult,
    RunStatus,
    StopReason,
)
from ejagent.contracts.session import (
    SessionCommit,
    SessionSnapshot,
    SessionStoreSerializationError,
)
from ejagent.contracts.usage import RunUsage


def session_commit_to_dict(commit: SessionCommit) -> dict[str, Any]:
    """Encode one idempotent commit into a stable JSON-compatible object."""

    return {
        "agent_id": commit.agent_id,
        "base": conversation_to_dict(commit.base),
        "outcome": _outcome_to_dict(commit.outcome),
    }


def session_commit_from_dict(value: Any) -> SessionCommit:
    """Decode and validate one stable SessionCommit object."""

    try:
        item = _mapping(value, "commit")
        return SessionCommit(
            agent_id=_string(item.get("agent_id"), "commit.agent_id"),
            base=conversation_from_dict(item.get("base"), label="commit.base"),
            outcome=_outcome_from_dict(item.get("outcome")),
        )
    except SessionStoreSerializationError:
        raise
    except (TypeError, ValueError) as exc:
        raise SessionStoreSerializationError(
            f"invalid SessionCommit payload: {exc}"
        ) from exc


def session_snapshot_to_dict(snapshot: SessionSnapshot) -> dict[str, Any]:
    """Encode Harness recovery state without Audit data."""

    return {
        "agent_id": snapshot.agent_id,
        "conversation": conversation_to_dict(snapshot.conversation),
        "last_result": (
            _result_to_dict(snapshot.last_result)
            if snapshot.last_result is not None
            else None
        ),
    }


def session_snapshot_from_dict(value: Any) -> SessionSnapshot:
    """Decode and validate Harness recovery state."""

    try:
        item = _mapping(value, "snapshot")
        raw_result = item.get("last_result")
        return SessionSnapshot(
            agent_id=_string(item.get("agent_id"), "snapshot.agent_id"),
            conversation=conversation_from_dict(
                item.get("conversation"),
                label="snapshot.conversation",
            ),
            last_result=(
                _result_from_dict(raw_result, label="snapshot.last_result")
                if raw_result is not None
                else None
            ),
        )
    except SessionStoreSerializationError:
        raise
    except (TypeError, ValueError) as exc:
        raise SessionStoreSerializationError(
            f"invalid SessionSnapshot payload: {exc}"
        ) from exc


def conversation_to_dict(conversation: ConversationSnapshot) -> dict[str, Any]:
    return {
        "revision": conversation.revision,
        "messages": [_message_to_dict(message) for message in conversation.messages],
    }


def conversation_from_dict(
    value: Any,
    *,
    label: str = "conversation",
) -> ConversationSnapshot:
    item = _mapping(value, label)
    messages = _list(item.get("messages"), f"{label}.messages")
    return ConversationSnapshot(
        revision=_integer(item.get("revision"), f"{label}.revision"),
        messages=tuple(
            _message_from_dict(message, f"{label}.messages[{index}]")
            for index, message in enumerate(messages)
        ),
    )


def run_audit_to_dict(audit: RunAudit) -> dict[str, Any]:
    return {
        "result": _result_to_dict(audit.result),
        "base_revision": audit.base_revision,
        "resulting_revision": audit.resulting_revision,
        "committed": audit.committed,
        "records": [_audit_record_to_dict(record) for record in audit.records],
        "failure": (
            _failure_to_dict(audit.failure) if audit.failure is not None else None
        ),
    }


def run_audit_from_dict(value: Any, *, label: str = "audit") -> RunAudit:
    try:
        item = _mapping(value, label)
        records = _list(item.get("records"), f"{label}.records")
        raw_failure = item.get("failure")
        return RunAudit(
            result=_result_from_dict(item.get("result"), label=f"{label}.result"),
            base_revision=_integer(
                item.get("base_revision"),
                f"{label}.base_revision",
            ),
            resulting_revision=_integer(
                item.get("resulting_revision"),
                f"{label}.resulting_revision",
            ),
            committed=_boolean(item.get("committed"), f"{label}.committed"),
            records=tuple(
                _audit_record_from_dict(record, f"{label}.records[{index}]")
                for index, record in enumerate(records)
            ),
            failure=(
                _failure_from_dict(raw_failure, label=f"{label}.failure")
                if raw_failure is not None
                else None
            ),
        )
    except SessionStoreSerializationError:
        raise
    except (TypeError, ValueError) as exc:
        raise SessionStoreSerializationError(
            f"invalid RunAudit payload: {exc}"
        ) from exc


def _message_to_dict(message: ConversationMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"type": "system", "content": message.content}
    if isinstance(message, UserMessage):
        return {"type": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        return {
            "type": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": thaw_json_value(call.arguments),
                }
                for call in message.tool_calls
            ],
        }
    if isinstance(message, ToolResultMessage):
        return {
            "type": "tool_result",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "result": thaw_json_value(message.result),
            "is_error": message.is_error,
        }
    raise TypeError(f"unsupported Conversation message {type(message).__name__}")


def _message_from_dict(value: Any, label: str) -> ConversationMessage:
    item = _mapping(value, label)
    kind = _string(item.get("type"), f"{label}.type")
    if kind == "system":
        return SystemMessage(_string(item.get("content"), f"{label}.content"))
    if kind == "user":
        return UserMessage(_string(item.get("content"), f"{label}.content"))
    if kind == "assistant":
        raw_calls = _list(item.get("tool_calls", []), f"{label}.tool_calls")
        content = _optional_string(item.get("content"), f"{label}.content")
        return AssistantMessage(
            content=content,
            tool_calls=tuple(
                _tool_call_from_dict(call, f"{label}.tool_calls[{index}]")
                for index, call in enumerate(raw_calls)
            ),
        )
    if kind == "tool_result":
        return ToolResultMessage(
            tool_call_id=_string(
                item.get("tool_call_id"),
                f"{label}.tool_call_id",
            ),
            tool_name=_string(item.get("tool_name"), f"{label}.tool_name"),
            result=item.get("result"),
            is_error=_boolean(item.get("is_error"), f"{label}.is_error"),
        )
    raise SessionStoreSerializationError(f"{label}.type has unsupported value {kind!r}")


def _tool_call_from_dict(value: Any, label: str) -> ToolCall:
    item = _mapping(value, label)
    arguments = _mapping(item.get("arguments"), f"{label}.arguments")
    return ToolCall(
        id=_string(item.get("id"), f"{label}.id"),
        name=_string(item.get("name"), f"{label}.name"),
        arguments=arguments,
    )


def _outcome_to_dict(outcome: RunOutcome) -> dict[str, Any]:
    return {
        "result": _result_to_dict(outcome.result),
        "delta": {
            "base_revision": outcome.delta.base_revision,
            "messages": [
                _message_to_dict(message) for message in outcome.delta.messages
            ],
        },
        "audit_records": [
            _audit_record_to_dict(record) for record in outcome.audit_records
        ],
        "failure": (
            _failure_to_dict(outcome.failure) if outcome.failure is not None else None
        ),
    }


def _outcome_from_dict(value: Any) -> RunOutcome:
    item = _mapping(value, "commit.outcome")
    delta = _mapping(item.get("delta"), "commit.outcome.delta")
    raw_messages = _list(
        delta.get("messages"),
        "commit.outcome.delta.messages",
    )
    raw_records = _list(
        item.get("audit_records"),
        "commit.outcome.audit_records",
    )
    raw_failure = item.get("failure")
    return RunOutcome(
        result=_result_from_dict(
            item.get("result"),
            label="commit.outcome.result",
        ),
        delta=RunDelta(
            base_revision=_integer(
                delta.get("base_revision"),
                "commit.outcome.delta.base_revision",
            ),
            messages=tuple(
                _message_from_dict(
                    message,
                    f"commit.outcome.delta.messages[{index}]",
                )
                for index, message in enumerate(raw_messages)
            ),
        ),
        audit_records=tuple(
            _audit_record_from_dict(
                record,
                f"commit.outcome.audit_records[{index}]",
            )
            for index, record in enumerate(raw_records)
        ),
        failure=(
            _failure_from_dict(raw_failure, label="commit.outcome.failure")
            if raw_failure is not None
            else None
        ),
    )


def _result_to_dict(result: RunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "stop_reason": result.stop_reason.value,
        "turns": result.turns,
        "output": result.output,
        "usage": result.usage.to_dict(),
    }


def _result_from_dict(value: Any, *, label: str) -> RunResult:
    item = _mapping(value, label)
    return RunResult(
        run_id=_string(item.get("run_id"), f"{label}.run_id"),
        status=RunStatus(_string(item.get("status"), f"{label}.status")),
        stop_reason=StopReason(
            _string(item.get("stop_reason"), f"{label}.stop_reason")
        ),
        turns=_integer(item.get("turns"), f"{label}.turns"),
        output=_optional_string(item.get("output"), f"{label}.output"),
        usage=_usage_from_dict(item.get("usage"), f"{label}.usage"),
    )


def _usage_from_dict(value: Any, label: str) -> RunUsage:
    item = _mapping(value, label)
    return RunUsage(
        input_tokens=_integer(item.get("input_tokens"), f"{label}.input_tokens"),
        output_tokens=_integer(
            item.get("output_tokens"),
            f"{label}.output_tokens",
        ),
        total_tokens=_integer(item.get("total_tokens"), f"{label}.total_tokens"),
        request_count=_integer(
            item.get("request_count"),
            f"{label}.request_count",
        ),
        reported_request_count=_integer(
            item.get("reported_request_count"),
            f"{label}.reported_request_count",
        ),
        cache_read_tokens=_optional_integer(
            item.get("cache_read_tokens"),
            f"{label}.cache_read_tokens",
        ),
        cache_write_tokens=_optional_integer(
            item.get("cache_write_tokens"),
            f"{label}.cache_write_tokens",
        ),
        reasoning_tokens=_optional_integer(
            item.get("reasoning_tokens"),
            f"{label}.reasoning_tokens",
        ),
    )


def _failure_to_dict(failure: RunFailure) -> dict[str, Any]:
    return {
        "phase": failure.phase.value,
        "code": failure.code.value,
        "message": failure.message,
        "retryable": failure.retryable,
    }


def _failure_from_dict(value: Any, *, label: str) -> RunFailure:
    item = _mapping(value, label)
    return RunFailure(
        phase=RunPhase(_string(item.get("phase"), f"{label}.phase")),
        code=FailureCode(_string(item.get("code"), f"{label}.code")),
        message=_string(item.get("message"), f"{label}.message"),
        retryable=_boolean(item.get("retryable"), f"{label}.retryable"),
    )


def _audit_record_to_dict(record: AuditRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "sequence": record.sequence,
        "kind": record.kind,
        "occurred_at": record.occurred_at.isoformat(),
        "payload": thaw_json_value(record.payload),
    }


def _audit_record_from_dict(value: Any, label: str) -> AuditRecord:
    item = _mapping(value, label)
    raw_time = _string(item.get("occurred_at"), f"{label}.occurred_at")
    try:
        occurred_at = datetime.fromisoformat(raw_time)
    except ValueError as exc:
        raise SessionStoreSerializationError(
            f"{label}.occurred_at must be an ISO datetime"
        ) from exc
    payload = _mapping(item.get("payload"), f"{label}.payload")
    return AuditRecord(
        run_id=_string(item.get("run_id"), f"{label}.run_id"),
        sequence=_integer(item.get("sequence"), f"{label}.sequence"),
        kind=_string(item.get("kind"), f"{label}.kind"),
        occurred_at=occurred_at,
        payload=payload,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SessionStoreSerializationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SessionStoreSerializationError(f"{label} keys must be strings")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SessionStoreSerializationError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SessionStoreSerializationError(f"{label} must be a string")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionStoreSerializationError(f"{label} must be an integer")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SessionStoreSerializationError(f"{label} must be a boolean")
    return value
