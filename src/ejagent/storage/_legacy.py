from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ejagent.contracts.runs import RunStatus, StopReason
from ejagent.contracts.session import SessionMigrationError
from ejagent.contracts.usage import RunUsage

_LEGACY_JOURNAL_VERSION = 1
_LEGACY_SESSION_VERSION = 1
_MAIN_BRANCH = "main"
_REMEDIATION = "repair or export the legacy JSONL Session before migration"


@dataclass(frozen=True, slots=True)
class LegacyRunResult:
    status: RunStatus
    stop_reason: StopReason
    turns: int
    output: str | None
    error: str | None
    usage: RunUsage


@dataclass(slots=True)
class LegacyRun:
    run_id: str
    result: LegacyRunResult | None = None


@dataclass(frozen=True, slots=True)
class LegacySessionData:
    session_id: str
    agent_id: str | None
    entries: tuple[Mapping[str, Any], ...]
    runs: tuple[LegacyRun, ...]


def load_legacy_session(
    root: Path,
    session_id: str,
) -> LegacySessionData | None:
    """Project the main branch of one legacy JSONL Session journal."""

    digest = sha256(session_id.encode("utf-8")).hexdigest()
    path = root / f"{digest}.jsonl"
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _migration_error(f"failed to read legacy Session {path}") from exc

    records: list[Mapping[str, Any]] = []
    expected_revision = 1
    record_ids: set[str] = set()
    for index, line in enumerate(content.splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            if records:
                break
            raise _migration_error(f"legacy Session {path} has no complete record")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _migration_error(
                f"legacy Session {path} has invalid JSON at line {index + 1}"
            ) from exc
        record = _mapping(value, f"journal line {index + 1}")
        if (
            _integer(record.get("journal_schema_version"), "journal version")
            != _LEGACY_JOURNAL_VERSION
        ):
            raise _migration_error("unsupported legacy journal schema version")
        revision = _integer(record.get("revision"), "journal revision")
        if revision != expected_revision:
            raise _migration_error("legacy journal revisions are not contiguous")
        expected_revision += 1
        record_id = _string(record.get("record_id"), "journal record_id")
        if record_id in record_ids:
            raise _migration_error("legacy journal repeats a record_id")
        record_ids.add(record_id)
        if _string(record.get("session_id"), "journal session_id") != session_id:
            raise _migration_error("legacy journal contains another session_id")
        records.append(record)

    if not records:
        return None
    return _project_main(records, session_id=session_id)


def _project_main(
    records: list[Mapping[str, Any]],
    *,
    session_id: str,
) -> LegacySessionData:
    agent_id: str | None = None
    entries: list[Mapping[str, Any]] = []
    runs: list[LegacyRun] = []
    previous_id: str | None = None

    for record in records:
        if _string(record.get("branch_id"), "journal branch_id") != _MAIN_BRANCH:
            continue
        parent = record.get("parent_id")
        if parent is not None and not isinstance(parent, str):
            raise _migration_error("legacy journal parent_id is invalid")
        if parent != previous_id:
            raise _migration_error("legacy main branch parent chain is broken")
        previous_id = _string(record.get("record_id"), "journal record_id")
        kind = _string(record.get("type"), "journal type")
        data = _mapping(record.get("data"), f"{kind} data")
        raw_agent = record.get("agent_id")
        record_agent = (
            None if raw_agent is None else _string(raw_agent, "journal agent_id")
        )

        if kind == "checkpoint":
            checkpoint = _decode_checkpoint(data, session_id=session_id)
            agent_id = checkpoint.agent_id
            entries = list(checkpoint.entries)
            runs = list(checkpoint.runs)
            if agent_id != record_agent:
                raise _migration_error("legacy checkpoint agent_id does not match")
            continue
        if record_agent is None:
            raise _migration_error(f"legacy {kind} record has no agent_id")
        if agent_id is None:
            agent_id = record_agent
        elif agent_id != record_agent:
            raise _migration_error("legacy journal contains multiple agent IDs")

        if kind == "run_started":
            run_id = _string(data.get("run_id"), "run_started run_id")
            _require_new_run(runs, run_id)
            task = _string(data.get("task"), "run_started task")
            runs.append(LegacyRun(run_id))
            entries.append({"role": "user", "content": task})
        elif kind == "run_continued":
            run_id = _string(data.get("run_id"), "run_continued run_id")
            _require_new_run(runs, run_id)
            runs.append(LegacyRun(run_id))
        elif kind == "message_appended":
            _require_open_run(runs, data)
            entries.append(_mapping(data.get("message"), "message_appended message"))
        elif kind == "messages_appended":
            _require_open_run(runs, data)
            messages = _list(data.get("messages"), "messages_appended messages")
            entries.extend(
                _mapping(message, f"messages_appended messages[{index}]")
                for index, message in enumerate(messages)
            )
        elif kind == "steering_applied":
            _require_open_run(runs, data)
            entries.append(
                {
                    "role": "user",
                    "content": _string(data.get("content"), "steering content"),
                }
            )
        elif kind == "run_finished":
            run = _require_open_run(runs, data)
            run.result = _decode_result(data.get("result"), "run_finished result")
        elif kind in {"compaction_applied", "branch_created"}:
            continue
        else:
            raise _migration_error(f"unsupported legacy record type {kind!r}")

    return LegacySessionData(
        session_id=session_id,
        agent_id=agent_id,
        entries=tuple(entries),
        runs=tuple(runs),
    )


def _decode_checkpoint(
    data: Mapping[str, Any],
    *,
    session_id: str,
) -> LegacySessionData:
    document = _mapping(data.get("document"), "checkpoint document")
    if (
        _integer(document.get("schema_version"), "session schema_version")
        != _LEGACY_SESSION_VERSION
    ):
        raise _migration_error("unsupported legacy Session schema version")
    payload = _mapping(document.get("session"), "checkpoint session")
    if _string(payload.get("session_id"), "checkpoint session_id") != session_id:
        raise _migration_error("legacy checkpoint session_id does not match")
    raw_agent = payload.get("agent_id")
    agent_id = None if raw_agent is None else _string(raw_agent, "checkpoint agent_id")
    raw_entries = _list(payload.get("entries"), "checkpoint entries")
    entries = tuple(
        _mapping(
            _mapping(item, f"checkpoint entries[{index}]").get("message"),
            f"checkpoint entries[{index}].message",
        )
        for index, item in enumerate(raw_entries)
    )
    raw_runs = _list(payload.get("runs"), "checkpoint runs")
    runs: list[LegacyRun] = []
    for index, value in enumerate(raw_runs):
        item = _mapping(value, f"checkpoint runs[{index}]")
        run_id = _string(item.get("run_id"), f"checkpoint runs[{index}].run_id")
        _require_new_run(runs, run_id)
        raw_result = item.get("result")
        runs.append(
            LegacyRun(
                run_id,
                None
                if raw_result is None
                else _decode_result(raw_result, f"checkpoint runs[{index}].result"),
            )
        )
    return LegacySessionData(session_id, agent_id, entries, tuple(runs))


def _decode_result(value: Any, label: str) -> LegacyRunResult:
    item = _mapping(value, label)
    return LegacyRunResult(
        status=RunStatus(_string(item.get("status"), f"{label}.status")),
        stop_reason=StopReason(
            _string(item.get("stop_reason"), f"{label}.stop_reason")
        ),
        turns=_integer(item.get("turns"), f"{label}.turns"),
        output=_optional_string(item.get("output"), f"{label}.output"),
        error=_optional_string(item.get("error"), f"{label}.error"),
        usage=_decode_usage(item.get("usage"), f"{label}.usage"),
    )


def _decode_usage(value: Any, label: str) -> RunUsage:
    item = _mapping(value, label)
    return RunUsage(
        input_tokens=_integer(item.get("input_tokens"), f"{label}.input_tokens"),
        output_tokens=_integer(item.get("output_tokens"), f"{label}.output_tokens"),
        total_tokens=_integer(item.get("total_tokens"), f"{label}.total_tokens"),
        request_count=_integer(item.get("request_count"), f"{label}.request_count"),
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


def _require_new_run(runs: list[LegacyRun], run_id: str) -> None:
    if any(run.run_id == run_id for run in runs):
        raise _migration_error(f"legacy journal repeats run_id {run_id!r}")
    if any(run.result is None for run in runs):
        raise _migration_error("legacy journal starts a Run before finishing another")


def _require_open_run(
    runs: list[LegacyRun],
    data: Mapping[str, Any],
) -> LegacyRun:
    run_id = _string(data.get("run_id"), "record run_id")
    for run in runs:
        if run.run_id == run_id:
            if run.result is not None:
                raise _migration_error(f"legacy Run {run_id!r} is already finished")
            return run
    raise _migration_error(f"legacy journal references unknown Run {run_id!r}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise _migration_error(f"{label} must be an object with string keys")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise _migration_error(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise _migration_error(f"{label} must be text")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _migration_error(f"{label} must be an integer")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _migration_error(message: str) -> SessionMigrationError:
    return SessionMigrationError(message, remediation=_REMEDIATION)
