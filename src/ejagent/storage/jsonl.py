from __future__ import annotations

import asyncio
import json
import math
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar

from ejagent.contracts.audit import AuditReader, RunAudit
from ejagent.contracts.session import (
    SessionCommit,
    SessionConflictError,
    SessionSnapshot,
    SessionStore,
    SessionStoreError,
    SessionStoreSerializationError,
)
from ejagent.storage._file_lock import StoreFileLock
from ejagent.storage.codec import (
    session_commit_from_dict,
    session_commit_to_dict,
)

STORE_SCHEMA_VERSION = 1
_ResultT = TypeVar("_ResultT")


@dataclass(slots=True)
class _StoreIndex:
    snapshot: SessionSnapshot | None
    audit: list[RunAudit]
    commits: dict[str, tuple[SessionCommit, SessionSnapshot]]
    record_count: int = 0
    valid_length: int = 0

    @classmethod
    def empty(cls) -> _StoreIndex:
        return cls(None, [], {})


class JsonlSessionStore(SessionStore, AuditReader):
    """Durable append-only SessionStore with cross-process compare-and-commit."""

    def __init__(
        self,
        root: str | Path,
        *,
        lock_timeout: float | None = 10.0,
    ) -> None:
        if lock_timeout is not None:
            if isinstance(lock_timeout, bool) or not isinstance(
                lock_timeout, (int, float)
            ):
                raise TypeError("lock_timeout must be a number or None")
            if lock_timeout < 0 or not math.isfinite(lock_timeout):
                raise ValueError("lock_timeout must be finite and non-negative")
            lock_timeout = float(lock_timeout)
        self.root = Path(root).expanduser()
        self.lock_timeout = lock_timeout
        self._lock = asyncio.Lock()

    async def load(self, agent_id: str) -> SessionSnapshot | None:
        normalized = self._normalize_agent_id(agent_id)
        path = self._path_for(normalized)
        async with self._lock:
            index = await self._read_index(path, normalized)
            return index.snapshot

    async def commit(self, commit: SessionCommit) -> SessionSnapshot:
        if not isinstance(commit, SessionCommit):
            raise TypeError("commit must be a SessionCommit")
        normalized = self._normalize_agent_id(commit.agent_id)
        path = self._path_for(normalized)
        self._validate_json(session_commit_to_dict(commit), label="SessionCommit")
        async with self._lock:
            return await self._run_locked(
                path,
                self._commit_sync,
                path,
                commit,
            )

    async def load_audit(self, agent_id: str) -> tuple[RunAudit, ...]:
        normalized = self._normalize_agent_id(agent_id)
        path = self._path_for(normalized)
        async with self._lock:
            index = await self._read_index(path, normalized)
            return tuple(index.audit)

    def _commit_sync(
        self,
        path: Path,
        commit: SessionCommit,
    ) -> SessionSnapshot:
        index = self._read_sync(path, commit.agent_id)
        existing = index.commits.get(commit.run_id)
        if existing is not None:
            previous, snapshot = existing
            if previous != commit:
                raise SessionConflictError(
                    f"run_id {commit.run_id!r} already identifies a different commit"
                )
            return snapshot
        current = index.snapshot
        if current is None:
            if commit.base_revision != 0:
                raise SessionConflictError(
                    f"agent {commit.agent_id!r} has no revision {commit.base_revision}"
                )
            current = SessionSnapshot(
                agent_id=commit.agent_id,
                conversation=commit.base,
            )
        self._validate_base(current, commit)
        snapshot = self._resulting_snapshot(current, commit)
        record = self._record(
            sequence=index.record_count + 1,
            agent_id=commit.agent_id,
            kind="commit",
            data={"commit": session_commit_to_dict(commit)},
        )
        self._write_record_sync(path, record, index.valid_length)
        return snapshot

    async def _read_index(self, path: Path, agent_id: str) -> _StoreIndex:
        return await self._run_worker(self._read_locked_sync, path, agent_id)

    async def _run_locked(
        self,
        path: Path,
        operation: Callable[..., _ResultT],
        *args: Any,
    ) -> _ResultT:
        return await self._run_worker(
            self._call_locked_sync,
            path,
            operation,
            args,
        )

    async def _run_worker(
        self,
        operation: Callable[..., _ResultT],
        *args: Any,
    ) -> _ResultT:
        cancelled = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(operation, *args, cancelled))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled.set()
            while True:
                try:
                    await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            raise

    def _read_locked_sync(
        self,
        path: Path,
        agent_id: str,
        cancelled: threading.Event,
    ) -> _StoreIndex:
        lock_path = self._lock_path_for(path)
        if not path.exists() and not lock_path.exists():
            return _StoreIndex.empty()
        return self._call_locked_sync(
            path,
            self._read_sync,
            (path, agent_id),
            cancelled,
        )

    def _call_locked_sync(
        self,
        path: Path,
        operation: Callable[..., _ResultT],
        args: tuple[Any, ...],
        cancelled: threading.Event,
    ) -> _ResultT:
        lock = StoreFileLock(self._lock_path_for(path), timeout=self.lock_timeout)
        with lock.acquire(cancelled):
            return operation(*args)

    def _read_sync(self, path: Path, agent_id: str) -> _StoreIndex:
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return _StoreIndex.empty()
        except OSError as exc:
            raise SessionStoreError(f"failed to read SessionStore {path}") from exc

        index = _StoreIndex.empty()
        lines = content.splitlines(keepends=True)
        for line_index, encoded_line in enumerate(lines):
            line_number = line_index + 1
            if not encoded_line.endswith(b"\n"):
                if line_index == len(lines) - 1:
                    break
            try:
                raw: Any = json.loads(encoded_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SessionStoreSerializationError(
                    f"SessionStore {path} has invalid JSON at line {line_number}"
                ) from exc
            record = self._parse_record(raw, path=path, line_number=line_number)
            if record["agent_id"] != agent_id:
                raise SessionStoreSerializationError(
                    f"SessionStore {path} line {line_number} contains agent "
                    f"{record['agent_id']!r}, expected {agent_id!r}"
                )
            if record["sequence"] != index.record_count + 1:
                raise SessionStoreSerializationError(
                    f"SessionStore sequence jumped at {path}:{line_number}"
                )
            self._apply_record(index, record, path=path, line_number=line_number)
            index.record_count += 1
            index.valid_length += len(encoded_line)
        return index

    def _apply_record(
        self,
        index: _StoreIndex,
        record: dict[str, Any],
        *,
        path: Path,
        line_number: int,
    ) -> None:
        kind = record["type"]
        data = record["data"]
        if kind != "commit":
            raise SessionStoreSerializationError(
                f"unsupported record type {kind!r} at {path}:{line_number}"
            )
        commit = session_commit_from_dict(data.get("commit"))
        if commit.agent_id != record["agent_id"]:
            raise SessionStoreSerializationError(
                f"commit agent mismatch at {path}:{line_number}"
            )
        if commit.run_id in index.commits:
            raise SessionStoreSerializationError(
                f"SessionStore repeats run_id {commit.run_id!r} at {path}:{line_number}"
            )
        current = index.snapshot
        if current is None:
            if commit.base_revision != 0:
                raise SessionStoreSerializationError(
                    f"first commit has nonzero base revision at {path}:{line_number}"
                )
            current = SessionSnapshot(
                agent_id=commit.agent_id,
                conversation=commit.base,
            )
        try:
            self._validate_base(current, commit)
        except SessionConflictError as exc:
            raise SessionStoreSerializationError(
                f"stale stored commit at {path}:{line_number}: {exc}"
            ) from exc
        snapshot = self._resulting_snapshot(current, commit)
        index.snapshot = snapshot
        index.audit.append(commit.audit)
        index.commits[commit.run_id] = (commit, snapshot)

    @staticmethod
    def _validate_base(current: SessionSnapshot, commit: SessionCommit) -> None:
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

    @staticmethod
    def _resulting_snapshot(
        current: SessionSnapshot,
        commit: SessionCommit,
    ) -> SessionSnapshot:
        return SessionSnapshot(
            agent_id=commit.agent_id,
            conversation=commit.resulting_conversation,
            last_result=(
                commit.outcome.result
                if commit.advances_revision
                else current.last_result
            ),
        )

    def _write_record_sync(
        self,
        path: Path,
        record: Mapping[str, Any],
        valid_length: int,
    ) -> None:
        self._validate_json(record, label="SessionStore record")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            line = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                current_size = os.fstat(descriptor).st_size
                if valid_length < current_size:
                    os.ftruncate(descriptor, valid_length)
                os.lseek(descriptor, 0, os.SEEK_END)
                written = os.write(descriptor, line)
                if written != len(line):
                    raise OSError(
                        f"partial SessionStore write: {written}/{len(line)} bytes"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._fsync_directory()
        except OSError as exc:
            raise SessionStoreError(f"failed to append SessionStore {path}") from exc

    @staticmethod
    def _record(
        *,
        sequence: int,
        agent_id: str,
        kind: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "store_schema_version": STORE_SCHEMA_VERSION,
            "sequence": sequence,
            "agent_id": agent_id,
            "type": kind,
            "data": dict(data),
        }

    @staticmethod
    def _parse_record(
        value: Any,
        *,
        path: Path,
        line_number: int,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise SessionStoreSerializationError(
                f"SessionStore {path} line {line_number} must be an object"
            )
        version = value.get("store_schema_version")
        if version != STORE_SCHEMA_VERSION:
            raise SessionStoreSerializationError(
                f"unsupported store_schema_version {version!r} "
                f"at {path}:{line_number}; expected {STORE_SCHEMA_VERSION}"
            )
        sequence = value.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise SessionStoreSerializationError(
                f"SessionStore sequence must be a positive integer "
                f"at {path}:{line_number}"
            )
        agent_id = value.get("agent_id")
        kind = value.get("type")
        data = value.get("data")
        if not isinstance(agent_id, str) or not agent_id:
            raise SessionStoreSerializationError(
                f"SessionStore agent_id is invalid at {path}:{line_number}"
            )
        if not isinstance(kind, str) or not kind:
            raise SessionStoreSerializationError(
                f"SessionStore type is invalid at {path}:{line_number}"
            )
        if not isinstance(data, dict):
            raise SessionStoreSerializationError(
                f"SessionStore data must be an object at {path}:{line_number}"
            )
        return {
            "sequence": sequence,
            "agent_id": agent_id,
            "type": kind,
            "data": data,
        }

    @staticmethod
    def _validate_json(value: Any, *, label: str) -> None:
        try:
            json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise SessionStoreSerializationError(
                f"{label} is not JSON-compatible: {exc}"
            ) from exc

    @staticmethod
    def _normalize_agent_id(agent_id: str) -> str:
        if not isinstance(agent_id, str):
            raise TypeError("agent_id must be a string")
        if not agent_id.strip():
            raise ValueError("agent_id must not be empty")
        return agent_id

    def _path_for(self, agent_id: str) -> Path:
        digest = sha256(agent_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.core.jsonl"

    @staticmethod
    def _lock_path_for(path: Path) -> Path:
        return path.with_name(f"{path.name}.lock")

    def _fsync_directory(self) -> None:
        try:
            descriptor = os.open(self.root, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)
