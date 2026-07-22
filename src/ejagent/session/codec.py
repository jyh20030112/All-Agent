from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ejagent.agent.compaction import SummaryEntry
from ejagent.agent.result import AgentRunResult, RunStatus, StopReason
from ejagent.agent.usage import RunUsage
from ejagent.session.errors import SessionSerializationError
from ejagent.session.types import (
    AgentSession,
    SessionCompaction,
    SessionMessage,
    SessionRun,
    SessionRunIntent,
)

SESSION_SCHEMA_VERSION = 1


def session_to_dict(session: AgentSession) -> dict[str, Any]:
    """Encode one detached Session into the stable versioned data shape."""

    snapshot = session.snapshot()
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session": {
            "session_id": snapshot.session_id,
            "agent_id": snapshot.agent_id,
            "entries": [
                {
                    "run_id": entry.run_id,
                    "sequence": entry.sequence,
                    "message": deepcopy(entry.message),
                }
                for entry in snapshot.entries
            ],
            "runs": [_run_to_dict(run) for run in snapshot.runs],
            "compactions": [
                {
                    "operation_id": compaction.operation_id,
                    "sequence": compaction.sequence,
                    "summary": compaction.summary.to_dict(),
                    "messages": deepcopy(list(compaction.messages)),
                    "covered_entry_count": compaction.covered_entry_count,
                }
                for compaction in snapshot.compactions
            ],
        },
    }


def session_from_dict(value: Mapping[str, Any]) -> AgentSession:
    """Decode and validate one versioned Session data structure."""

    try:
        root = _mapping(value, "session document")
        version = _integer(root.get("schema_version"), "schema_version")
        if version != SESSION_SCHEMA_VERSION:
            raise SessionSerializationError(
                f"unsupported session schema_version {version}; "
                f"expected {SESSION_SCHEMA_VERSION}"
            )
        payload = _mapping(root.get("session"), "session")
        session_id = _string(payload.get("session_id"), "session.session_id")
        agent_id = _optional_string(
            payload.get("agent_id"),
            "session.agent_id",
        )
        entries = [
            _entry_from_dict(item, index)
            for index, item in enumerate(
                _list(payload.get("entries"), "session.entries")
            )
        ]
        runs = [
            _run_from_dict(item, index)
            for index, item in enumerate(_list(payload.get("runs"), "session.runs"))
        ]
        compactions = [
            _compaction_from_dict(item, index)
            for index, item in enumerate(
                _list(payload.get("compactions"), "session.compactions")
            )
        ]
        return AgentSession(
            session_id=session_id,
            agent_id=agent_id,
            entries=entries,
            runs=runs,
            compactions=compactions,
        )
    except SessionSerializationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionSerializationError(f"invalid session payload: {exc}") from exc


def _run_to_dict(run: SessionRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "task": run.task,
        "intent": run.intent.value,
        "start_sequence": run.start_sequence,
        "finish_sequence": run.finish_sequence,
        "result": (
            agent_run_result_to_dict(run.result) if run.result is not None else None
        ),
    }


def agent_run_result_to_dict(result: AgentRunResult) -> dict[str, Any]:
    """Encode a structured Run result for Session journal records."""

    return {
        "status": result.status.value,
        "stop_reason": result.stop_reason.value,
        "turns": result.turns,
        "output": result.output,
        "error": result.error,
        "usage": result.usage.to_dict(),
    }


def _entry_from_dict(value: Any, index: int) -> SessionMessage:
    label = f"session.entries[{index}]"
    item = _mapping(value, label)
    return SessionMessage(
        run_id=_string(item.get("run_id"), f"{label}.run_id"),
        sequence=_integer(item.get("sequence"), f"{label}.sequence"),
        message=_message(item.get("message"), f"{label}.message"),
    )


def _run_from_dict(value: Any, index: int) -> SessionRun:
    label = f"session.runs[{index}]"
    item = _mapping(value, label)
    raw_result = item.get("result")
    result = (
        None
        if raw_result is None
        else agent_run_result_from_dict(raw_result, label=label)
    )
    finish_sequence = _optional_integer(
        item.get("finish_sequence"),
        f"{label}.finish_sequence",
    )
    return SessionRun(
        run_id=_string(item.get("run_id"), f"{label}.run_id"),
        task=_optional_string(item.get("task"), f"{label}.task"),
        start_sequence=_integer(
            item.get("start_sequence"),
            f"{label}.start_sequence",
        ),
        intent=SessionRunIntent(_string(item.get("intent", "task"), f"{label}.intent")),
        finish_sequence=finish_sequence,
        result=result,
    )


def agent_run_result_from_dict(
    value: Any,
    *,
    label: str = "result",
) -> AgentRunResult:
    """Decode a structured Run result from a Session journal record."""

    label = f"{label}.result" if not label.endswith(".result") else label
    item = _mapping(value, label)
    return AgentRunResult(
        status=RunStatus(_string(item.get("status"), f"{label}.status")),
        stop_reason=StopReason(
            _string(item.get("stop_reason"), f"{label}.stop_reason")
        ),
        turns=_integer(item.get("turns"), f"{label}.turns"),
        output=_optional_string(item.get("output"), f"{label}.output"),
        error=_optional_string(item.get("error"), f"{label}.error"),
        usage=_usage_from_dict(item.get("usage"), label),
    )


def _usage_from_dict(value: Any, parent_label: str) -> RunUsage:
    label = f"{parent_label}.usage"
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


def _compaction_from_dict(value: Any, index: int) -> SessionCompaction:
    label = f"session.compactions[{index}]"
    item = _mapping(value, label)
    raw_messages = _list(item.get("messages"), f"{label}.messages")
    summary_value = _mapping(item.get("summary"), f"{label}.summary")
    return SessionCompaction(
        operation_id=_string(
            item.get("operation_id"),
            f"{label}.operation_id",
        ),
        sequence=_integer(item.get("sequence"), f"{label}.sequence"),
        summary=SummaryEntry.from_dict(summary_value),
        messages=tuple(
            _message(message, f"{label}.messages[{message_index}]")
            for message_index, message in enumerate(raw_messages)
        ),
        covered_entry_count=_integer(
            item.get("covered_entry_count"),
            f"{label}.covered_entry_count",
        ),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SessionSerializationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SessionSerializationError(f"{label} keys must be strings")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SessionSerializationError(f"{label} must be an array")
    return value


def _message(value: Any, label: str) -> dict[str, Any]:
    return deepcopy(dict(_mapping(value, label)))


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SessionSerializationError(f"{label} must be a string")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SessionSerializationError(f"{label} must be an integer")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)
