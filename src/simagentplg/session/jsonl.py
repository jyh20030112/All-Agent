from __future__ import annotations

import asyncio
import json
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from simagentplg.session._file_lock import SessionFileLock
from simagentplg.session.errors import (
    SessionConflictError,
    SessionSerializationError,
    SessionStorageError,
)
from simagentplg.session.journal import (
    DEFAULT_SESSION_BRANCH,
    SessionRecord,
    SessionRecordDraft,
    SessionRecordKind,
    apply_session_record,
)
from simagentplg.session.tree import (
    SessionBranch,
    SessionBranchIntent,
    SessionCheckout,
    SessionRetry,
)
from simagentplg.session.types import AgentSession

_ResultT = TypeVar("_ResultT")


@dataclass(slots=True)
class _JournalIndex:
    records: list[SessionRecord]
    records_by_id: dict[str, SessionRecord]
    branches: dict[str, SessionBranch]
    valid_length: int = 0

    @classmethod
    def empty(cls) -> _JournalIndex:
        return cls(records=[], records_by_id={}, branches={})


class JsonlSessionStorage:
    """Append and project immutable, tree-addressable Session records.

    One JSONL file contains every branch for a Session. File order assigns a
    global revision while ``parent_id`` defines the logical tree. A stable
    sidecar lock makes read-validate-append atomic across instances and
    processes on POSIX filesystems.
    """

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

    async def load(self, session_id: str) -> AgentSession | None:
        """Load the detached ``main`` branch projection for compatibility."""

        checkout = await self.checkout(session_id)
        return checkout.session if checkout is not None else None

    async def checkout(
        self,
        session_id: str,
        *,
        branch_id: str = DEFAULT_SESSION_BRANCH,
        record_id: str | None = None,
    ) -> SessionCheckout | None:
        """Project one branch head or an exact record without changing history."""

        normalized_id = self._normalize_session_id(session_id)
        normalized_branch = self._normalize_branch_id(branch_id)
        path = self._path_for(normalized_id)
        async with self._lock:
            index = await self._read_index(path, normalized_id)
            return self._checkout_sync(
                index,
                normalized_id,
                branch_id=normalized_branch,
                record_id=record_id,
            )

    async def head(
        self,
        session_id: str,
        *,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> SessionRecord | None:
        """Return the immutable record currently heading one branch."""

        normalized_id = self._normalize_session_id(session_id)
        normalized_branch = self._normalize_branch_id(branch_id)
        path = self._path_for(normalized_id)
        async with self._lock:
            index = await self._read_index(path, normalized_id)
            branch = index.branches.get(normalized_branch)
            if branch is None:
                return None
            return index.records_by_id[branch.head_record_id]

    async def list_branches(self, session_id: str) -> tuple[SessionBranch, ...]:
        """Return branches ordered by their creation revision."""

        normalized_id = self._normalize_session_id(session_id)
        path = self._path_for(normalized_id)
        async with self._lock:
            index = await self._read_index(path, normalized_id)
            return tuple(
                sorted(
                    index.branches.values(), key=lambda branch: branch.created_revision
                )
            )

    async def save(self, session: AgentSession) -> None:
        """Append a full logical Checkpoint to the ``main`` branch."""

        await self.append(SessionRecordDraft.checkpoint(session.snapshot()))

    async def append(
        self,
        draft: SessionRecordDraft,
        *,
        expected_head_id: str | None = None,
        check_head: bool = False,
    ) -> SessionRecord:
        """Append one branch mutation, optionally comparing its current head."""

        if draft.kind is SessionRecordKind.BRANCH_CREATED:
            raise ValueError("use fork(), rollback(), or prepare_retry()")
        self._validate_draft_json(draft)
        path = self._path_for(draft.session_id)
        async with self._lock:
            return await self._run_locked(
                path,
                self._append_sync,
                path,
                draft,
                expected_head_id,
                check_head,
            )

    async def records(self, session_id: str) -> tuple[SessionRecord, ...]:
        """Return detached immutable records in physical append order."""

        normalized_id = self._normalize_session_id(session_id)
        path = self._path_for(normalized_id)
        async with self._lock:
            index = await self._read_index(path, normalized_id)
            return tuple(index.records)

    async def fork(
        self,
        session_id: str,
        *,
        source_branch: str = DEFAULT_SESSION_BRANCH,
        from_record_id: str | None = None,
        branch_id: str | None = None,
    ) -> SessionCheckout:
        """Create a general-purpose branch at a completed Session projection."""

        return await self._create_branch(
            session_id,
            source_branch=source_branch,
            base_record_id=from_record_id,
            branch_id=branch_id,
            intent=SessionBranchIntent.FORK,
        )

    async def rollback(
        self,
        session_id: str,
        *,
        to_record_id: str,
        source_branch: str = DEFAULT_SESSION_BRANCH,
        branch_id: str | None = None,
    ) -> SessionCheckout:
        """Create a branch at an ancestor without rewriting the source branch."""

        if not to_record_id:
            raise ValueError("to_record_id must not be empty")
        return await self._create_branch(
            session_id,
            source_branch=source_branch,
            base_record_id=to_record_id,
            branch_id=branch_id,
            intent=SessionBranchIntent.ROLLBACK,
        )

    async def prepare_retry(
        self,
        session_id: str,
        *,
        run_id: str,
        source_branch: str = DEFAULT_SESSION_BRANCH,
        branch_id: str | None = None,
    ) -> SessionRetry:
        """Branch before one Run and return its original task for explicit retry."""

        run_id = run_id.strip()
        if not run_id:
            raise ValueError("run_id must not be empty")
        normalized_id = self._normalize_session_id(session_id)
        normalized_source = self._normalize_branch_id(source_branch)
        normalized_branch = (
            self._normalize_branch_id(branch_id)
            if branch_id is not None
            else self._generated_branch_id(SessionBranchIntent.RETRY)
        )
        path = self._path_for(normalized_id)
        async with self._lock:
            return await self._run_locked(
                path,
                self._prepare_retry_sync,
                path,
                normalized_id,
                run_id,
                normalized_source,
                normalized_branch,
            )

    async def _create_branch(
        self,
        session_id: str,
        *,
        source_branch: str,
        base_record_id: str | None,
        branch_id: str | None,
        intent: SessionBranchIntent,
    ) -> SessionCheckout:
        normalized_id = self._normalize_session_id(session_id)
        normalized_source = self._normalize_branch_id(source_branch)
        normalized_base = (
            self._normalize_record_id(base_record_id)
            if base_record_id is not None
            else None
        )
        normalized_branch = (
            self._normalize_branch_id(branch_id)
            if branch_id is not None
            else self._generated_branch_id(intent)
        )
        path = self._path_for(normalized_id)
        async with self._lock:
            return await self._run_locked(
                path,
                self._create_branch_sync,
                path,
                normalized_id,
                normalized_source,
                normalized_base,
                normalized_branch,
                intent,
                None,
            )

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id must not be empty")
        return normalized

    @staticmethod
    def _normalize_branch_id(branch_id: str) -> str:
        if not isinstance(branch_id, str):
            raise TypeError("branch_id must be a string")
        normalized = branch_id.strip()
        if not normalized:
            raise ValueError("branch_id must not be empty")
        return normalized

    @staticmethod
    def _normalize_record_id(record_id: str) -> str:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        normalized = record_id.strip()
        if not normalized:
            raise ValueError("record_id must not be empty")
        return normalized

    @staticmethod
    def _generated_branch_id(intent: SessionBranchIntent) -> str:
        return f"{intent.value}-{uuid4().hex[:12]}"

    def _path_for(self, session_id: str) -> Path:
        normalized = self._normalize_session_id(session_id)
        digest = sha256(normalized.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.jsonl"

    @staticmethod
    def _lock_path_for(path: Path) -> Path:
        return path.with_name(f"{path.name}.lock")

    async def _read_index(
        self,
        path: Path,
        expected_session_id: str,
    ) -> _JournalIndex:
        return await self._run_worker(
            self._read_locked_sync,
            path,
            expected_session_id,
        )

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
        expected_session_id: str,
        cancelled: threading.Event,
    ) -> _JournalIndex:
        lock_path = self._lock_path_for(path)
        if not path.exists() and not lock_path.exists():
            return _JournalIndex.empty()
        return self._call_locked_sync(
            path,
            self._read_sync,
            (path, expected_session_id),
            cancelled,
        )

    def _call_locked_sync(
        self,
        path: Path,
        operation: Callable[..., _ResultT],
        args: tuple[Any, ...],
        cancelled: threading.Event,
    ) -> _ResultT:
        lock = SessionFileLock(
            self._lock_path_for(path),
            timeout=self.lock_timeout,
        )
        with lock.acquire(cancelled):
            return operation(*args)

    @staticmethod
    def _validate_draft_json(draft: SessionRecordDraft) -> None:
        try:
            json.dumps(draft.data, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise SessionSerializationError(
                f"Session record is not JSON-compatible: {exc}"
            ) from exc

    def _append_sync(
        self,
        path: Path,
        draft: SessionRecordDraft,
        expected_head_id: str | None,
        check_head: bool,
    ) -> SessionRecord:
        index = self._read_sync(path, draft.session_id)
        branch = index.branches.get(draft.branch_id)
        if branch is None and draft.branch_id != DEFAULT_SESSION_BRANCH:
            raise ValueError(f"unknown Session branch {draft.branch_id!r}")
        current_head = branch.head_record_id if branch is not None else None
        if check_head and current_head != expected_head_id:
            raise SessionConflictError(
                f"Session branch {draft.branch_id!r} head changed from "
                f"{expected_head_id!r} to {current_head!r}"
            )
        record = SessionRecord(
            record_id=uuid4().hex,
            parent_id=current_head,
            branch_id=draft.branch_id,
            revision=len(index.records) + 1,
            session_id=draft.session_id,
            agent_id=draft.agent_id,
            sequence=draft.sequence,
            kind=draft.kind,
            data=draft.data,
        )
        self._validate_candidate(index, record, path)
        self._write_record_sync(path, record, index.valid_length)
        return record

    def _create_branch_sync(
        self,
        path: Path,
        session_id: str,
        source_branch_id: str,
        base_record_id: str | None,
        branch_id: str,
        intent: SessionBranchIntent,
        retried_run_id: str | None,
    ) -> SessionCheckout:
        index = self._read_sync(path, session_id)
        source = self._require_branch(index, source_branch_id)
        if branch_id in index.branches:
            raise ValueError(f"Session branch {branch_id!r} already exists")
        base_id = base_record_id or source.head_record_id
        if base_id not in index.records_by_id:
            raise ValueError(f"unknown Session record {base_id!r}")
        if not self._is_ancestor(index, base_id, source.head_record_id):
            raise ValueError(
                f"record {base_id!r} is not an ancestor of branch {source_branch_id!r}"
            )
        base_session = self._project(index, base_id, session_id)
        self._require_finished(base_session, label=f"record {base_id!r}")
        draft = SessionRecordDraft.branch_created(
            session_id=session_id,
            agent_id=base_session.agent_id,
            branch_id=branch_id,
            base_record_id=base_id,
            source_branch_id=source_branch_id,
            source_head_id=source.head_record_id,
            intent=intent.value,
            retried_run_id=retried_run_id,
        )
        record = SessionRecord(
            record_id=uuid4().hex,
            parent_id=base_id,
            branch_id=branch_id,
            revision=len(index.records) + 1,
            session_id=session_id,
            agent_id=draft.agent_id,
            sequence=0,
            kind=SessionRecordKind.BRANCH_CREATED,
            data=draft.data,
        )
        self._validate_candidate(index, record, path)
        self._write_record_sync(path, record, index.valid_length)
        checkout = self._checkout_sync(index, session_id, branch_id=branch_id)
        if checkout is None:
            raise RuntimeError("created Session branch is unavailable")
        return checkout

    def _prepare_retry_sync(
        self,
        path: Path,
        session_id: str,
        run_id: str,
        source_branch_id: str,
        branch_id: str,
    ) -> SessionRetry:
        index = self._read_sync(path, session_id)
        source = self._require_branch(index, source_branch_id)
        if branch_id in index.branches:
            raise ValueError(f"Session branch {branch_id!r} already exists")
        run_record = self._find_run_started(index, source.head_record_id, run_id)
        if run_record is None:
            raise ValueError(
                f"unknown run {run_id!r} on Session branch {source_branch_id!r}"
            )
        task = run_record.data.get("task")
        if not isinstance(task, str) or not task:
            raise SessionSerializationError(
                f"run_started record {run_record.record_id!r} has no valid task"
            )
        if run_record.parent_id is None:
            base_session = AgentSession(session_id=session_id)
            if run_record.agent_id is not None:
                base_session.bind_agent(run_record.agent_id)
            checkout = self._create_root_retry_sync(
                path,
                index,
                base_session,
                source,
                branch_id,
                run_id,
            )
        else:
            checkout = self._create_branch_sync(
                path,
                session_id,
                source_branch_id,
                run_record.parent_id,
                branch_id,
                SessionBranchIntent.RETRY,
                run_id,
            )
        return SessionRetry(checkout=checkout, task=task, retried_run_id=run_id)

    def _create_root_retry_sync(
        self,
        path: Path,
        index: _JournalIndex,
        base_session: AgentSession,
        source: SessionBranch,
        branch_id: str,
        run_id: str,
    ) -> SessionCheckout:
        draft = SessionRecordDraft.branch_created(
            session_id=base_session.session_id,
            agent_id=base_session.agent_id,
            branch_id=branch_id,
            base_record_id=None,
            source_branch_id=source.branch_id,
            source_head_id=source.head_record_id,
            intent=SessionBranchIntent.RETRY.value,
            retried_run_id=run_id,
        )
        record = SessionRecord(
            record_id=uuid4().hex,
            parent_id=None,
            branch_id=branch_id,
            revision=len(index.records) + 1,
            session_id=base_session.session_id,
            agent_id=base_session.agent_id,
            sequence=0,
            kind=SessionRecordKind.BRANCH_CREATED,
            data=draft.data,
        )
        self._validate_candidate(index, record, path)
        self._write_record_sync(path, record, index.valid_length)
        checkout = self._checkout_sync(
            index,
            base_session.session_id,
            branch_id=branch_id,
        )
        if checkout is None:
            raise RuntimeError("created retry branch is unavailable")
        return checkout

    def _validate_candidate(
        self,
        index: _JournalIndex,
        record: SessionRecord,
        path: Path,
    ) -> None:
        self._add_record(index, record, path=path, line_number=record.revision)
        base = (
            self._project(index, record.parent_id, record.session_id)
            if record.parent_id is not None
            else None
        )
        self._apply_checked(base, record, path=path, line_number=record.revision)

    def _write_record_sync(
        self,
        path: Path,
        record: SessionRecord,
        valid_length: int,
    ) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionStorageError(
                f"failed to create Session journal directory {self.root}"
            ) from exc
        try:
            line = (
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SessionSerializationError(
                f"Session record is not JSON-compatible: {exc}"
            ) from exc

        descriptor: int | None = None
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
            current_size = os.fstat(descriptor).st_size
            if valid_length < current_size:
                os.ftruncate(descriptor, valid_length)
            os.lseek(descriptor, 0, os.SEEK_END)
            written = os.write(descriptor, line)
            if written != len(line):
                raise OSError(
                    f"partial Session journal write: {written}/{len(line)} bytes"
                )
            os.fsync(descriptor)
        except OSError as exc:
            raise SessionStorageError(
                f"failed to append Session journal {path}"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        self._fsync_directory()

    def _read_sync(self, path: Path, expected_session_id: str) -> _JournalIndex:
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return _JournalIndex.empty()
        except OSError as exc:
            raise SessionStorageError(f"failed to read Session journal {path}") from exc

        index = _JournalIndex.empty()
        branch_sessions: dict[str, AgentSession] = {}
        lines = content.splitlines(keepends=True)
        for line_index, encoded_line in enumerate(lines):
            line_number = line_index + 1
            if not encoded_line.endswith(b"\n"):
                if index.records and line_index == len(lines) - 1:
                    break
                raise SessionSerializationError(
                    f"Session journal {path} has an incomplete first record"
                )
            try:
                raw: Any = json.loads(encoded_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SessionSerializationError(
                    f"Session journal {path} has invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(raw, dict):
                raise SessionSerializationError(
                    f"Session journal {path} line {line_number} must be an object"
                )
            record = SessionRecord.from_dict(raw)
            if record.session_id != expected_session_id:
                raise SessionSerializationError(
                    f"Session journal {path} line {line_number} contains id "
                    f"{record.session_id!r}, expected {expected_session_id!r}"
                )
            previous_session = branch_sessions.get(record.branch_id)
            self._add_record(index, record, path=path, line_number=line_number)
            if record.kind is SessionRecordKind.BRANCH_CREATED:
                base = (
                    self._project(index, record.parent_id, expected_session_id)
                    if record.parent_id is not None
                    else None
                )
            else:
                base = previous_session
            branch_sessions[record.branch_id] = self._apply_checked(
                base,
                record,
                path=path,
                line_number=line_number,
            )
            index.valid_length += len(encoded_line)
        return index

    def _add_record(
        self,
        index: _JournalIndex,
        record: SessionRecord,
        *,
        path: Path,
        line_number: int,
    ) -> None:
        if record.record_id in index.records_by_id:
            raise SessionSerializationError(
                f"Session journal {path} repeats record_id "
                f"{record.record_id!r} at line {line_number}"
            )
        if record.revision != len(index.records) + 1:
            raise SessionSerializationError(
                f"Session journal revision jumped at {path}:{line_number}"
            )
        if record.parent_id is not None and record.parent_id not in index.records_by_id:
            raise SessionSerializationError(
                f"Session journal parent changed or is missing at {path}:{line_number}"
            )

        existing = index.branches.get(record.branch_id)
        if record.kind is SessionRecordKind.BRANCH_CREATED:
            branch = self._branch_from_record(index, record, path, line_number)
        elif existing is None:
            if record.branch_id != DEFAULT_SESSION_BRANCH:
                raise SessionSerializationError(
                    f"Session branch {record.branch_id!r} was not created"
                )
            if record.parent_id is not None:
                raise SessionSerializationError(
                    f"initial main record has a parent at {path}:{line_number}"
                )
            branch = SessionBranch(
                branch_id=DEFAULT_SESSION_BRANCH,
                head_record_id=record.record_id,
                base_record_id=None,
                source_branch_id=None,
                intent=None,
                created_revision=record.revision,
            )
        else:
            if record.parent_id != existing.head_record_id:
                raise SessionSerializationError(
                    f"Session branch head changed at {path}:{line_number}"
                )
            branch = replace(existing, head_record_id=record.record_id)

        index.records.append(record)
        index.records_by_id[record.record_id] = record
        index.branches[record.branch_id] = branch

    def _branch_from_record(
        self,
        index: _JournalIndex,
        record: SessionRecord,
        path: Path,
        line_number: int,
    ) -> SessionBranch:
        if record.branch_id == DEFAULT_SESSION_BRANCH:
            raise SessionSerializationError(
                "main branch must not be created explicitly"
            )
        if record.branch_id in index.branches:
            raise SessionSerializationError(
                f"Session branch {record.branch_id!r} already exists"
            )
        base_record_id = self._optional_data_string(record, "base_record_id")
        source_branch_id = self._required_data_string(record, "source_branch_id")
        source_head_id = self._required_data_string(record, "source_head_id")
        if record.parent_id != base_record_id:
            raise SessionSerializationError(
                f"branch base does not match parent at {path}:{line_number}"
            )
        source = index.branches.get(source_branch_id)
        if source is None:
            raise SessionSerializationError(
                f"unknown source branch {source_branch_id!r} at {path}:{line_number}"
            )
        if source.head_record_id != source_head_id:
            raise SessionSerializationError(
                f"source branch head does not match at {path}:{line_number}"
            )
        if base_record_id is not None and not self._is_ancestor(
            index,
            base_record_id,
            source_head_id,
        ):
            raise SessionSerializationError(
                f"branch base is not a source ancestor at {path}:{line_number}"
            )
        try:
            intent = SessionBranchIntent(self._required_data_string(record, "intent"))
        except ValueError as exc:
            raise SessionSerializationError(
                f"invalid branch intent at {path}:{line_number}"
            ) from exc
        retried_run_id = record.data.get("retried_run_id")
        if intent is SessionBranchIntent.RETRY:
            if not isinstance(retried_run_id, str) or not retried_run_id:
                raise SessionSerializationError(
                    f"retry branch has no retried_run_id at {path}:{line_number}"
                )
        elif retried_run_id is not None:
            raise SessionSerializationError(
                f"non-retry branch has retried_run_id at {path}:{line_number}"
            )
        if base_record_id is None and intent is not SessionBranchIntent.RETRY:
            raise SessionSerializationError(
                f"only a first-Run retry may branch from the root at "
                f"{path}:{line_number}"
            )
        return SessionBranch(
            branch_id=record.branch_id,
            head_record_id=record.record_id,
            base_record_id=base_record_id,
            source_branch_id=source_branch_id,
            intent=intent,
            created_revision=record.revision,
        )

    @staticmethod
    def _apply_checked(
        session: AgentSession | None,
        record: SessionRecord,
        *,
        path: Path,
        line_number: int,
    ) -> AgentSession:
        if (
            session is not None
            and session.agent_id is not None
            and record.agent_id != session.agent_id
        ):
            raise SessionSerializationError(
                f"Session journal {path} changes agent_id at line {line_number}"
            )
        try:
            return apply_session_record(session, record)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, SessionSerializationError):
                raise
            raise SessionSerializationError(
                f"invalid Session mutation at {path}:{line_number}: {exc}"
            ) from exc

    def _checkout_sync(
        self,
        index: _JournalIndex,
        session_id: str,
        *,
        branch_id: str,
        record_id: str | None = None,
    ) -> SessionCheckout | None:
        if record_id is None:
            branch = index.branches.get(branch_id)
            if branch is None:
                return None
            target_id = branch.head_record_id
        else:
            target = index.records_by_id.get(record_id)
            if target is None:
                raise ValueError(f"unknown Session record {record_id!r}")
            stored_branch = index.branches[target.branch_id]
            branch = replace(stored_branch, head_record_id=target.record_id)
            target_id = target.record_id
        session = self._project(index, target_id, session_id)
        return SessionCheckout(
            session=session,
            branch=branch,
            head=index.records_by_id[target_id],
        )

    def _project(
        self,
        index: _JournalIndex,
        record_id: str,
        session_id: str,
    ) -> AgentSession:
        path: list[SessionRecord] = []
        current_id: str | None = record_id
        while current_id is not None:
            record = index.records_by_id.get(current_id)
            if record is None:
                raise SessionSerializationError(
                    f"Session tree references missing record {current_id!r}"
                )
            path.append(record)
            current_id = record.parent_id
        session: AgentSession | None = None
        for record in reversed(path):
            session = apply_session_record(session, record)
        if session is None:
            return AgentSession(session_id=session_id)
        return session.snapshot()

    @staticmethod
    def _require_branch(index: _JournalIndex, branch_id: str) -> SessionBranch:
        branch = index.branches.get(branch_id)
        if branch is None:
            raise ValueError(f"unknown Session branch {branch_id!r}")
        return branch

    @staticmethod
    def _require_finished(session: AgentSession, *, label: str) -> None:
        unfinished = [run.run_id for run in session.runs if not run.finished]
        if unfinished:
            raise ValueError(
                f"cannot branch from {label} with unfinished run(s): "
                + ", ".join(unfinished)
            )

    @staticmethod
    def _is_ancestor(
        index: _JournalIndex,
        ancestor_id: str,
        descendant_id: str,
    ) -> bool:
        current_id: str | None = descendant_id
        while current_id is not None:
            if current_id == ancestor_id:
                return True
            current_id = index.records_by_id[current_id].parent_id
        return False

    @staticmethod
    def _find_run_started(
        index: _JournalIndex,
        head_record_id: str,
        run_id: str,
    ) -> SessionRecord | None:
        current_id: str | None = head_record_id
        while current_id is not None:
            record = index.records_by_id[current_id]
            if (
                record.kind is SessionRecordKind.RUN_STARTED
                and record.data.get("run_id") == run_id
            ):
                return record
            current_id = record.parent_id
        return None

    @staticmethod
    def _required_data_string(record: SessionRecord, key: str) -> str:
        value = record.data.get(key)
        if not isinstance(value, str) or not value:
            raise SessionSerializationError(
                f"branch_created data.{key} must be a non-empty string"
            )
        return value

    @staticmethod
    def _optional_data_string(record: SessionRecord, key: str) -> str | None:
        value = record.data.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise SessionSerializationError(
                f"branch_created data.{key} must be null or a non-empty string"
            )
        return value

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
