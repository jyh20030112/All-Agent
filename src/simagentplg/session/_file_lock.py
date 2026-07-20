from __future__ import annotations

import errno
import importlib
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simagentplg.session.errors import (
    SessionLockTimeoutError,
    SessionStorageError,
)

_fcntl: Any | None
try:
    _fcntl = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    _fcntl = None


class _SessionLockCancelled(Exception):
    """Internal signal used to stop a worker still waiting for a file lock."""


@dataclass(slots=True)
class _ProcessLockEntry:
    lock: threading.Lock
    users: int = 0


_PROCESS_LOCKS: dict[str, _ProcessLockEntry] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class SessionFileLock:
    """Process-local plus POSIX advisory lock for one stable sidecar file."""

    def __init__(
        self,
        path: Path,
        *,
        timeout: float | None,
        poll_interval: float = 0.01,
    ) -> None:
        self.path = path
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._key = os.path.realpath(path)

    @contextmanager
    def acquire(
        self,
        cancelled: threading.Event | None = None,
    ) -> Iterator[None]:
        """Acquire both lock layers within one shared timeout budget."""

        fcntl = _fcntl
        if fcntl is None:
            raise SessionStorageError(
                "JsonlSessionStorage cross-process locking requires a POSIX platform"
            )
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        entry = self._retain_process_lock()
        process_acquired = False
        descriptor: int | None = None
        file_acquired = False
        try:
            self._acquire_process_lock(entry, deadline, cancelled)
            process_acquired = True
            self._raise_if_cancelled(cancelled)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    self.path,
                    os.O_RDWR | os.O_CREAT,
                    0o600,
                )
            except OSError as exc:
                raise SessionStorageError(
                    f"failed to open Session journal lock {self.path}"
                ) from exc
            self._acquire_file_lock(descriptor, deadline, cancelled, fcntl)
            file_acquired = True
            self._raise_if_cancelled(cancelled)
            yield
        finally:
            if file_acquired and descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            if descriptor is not None:
                os.close(descriptor)
            if process_acquired:
                entry.lock.release()
            self._release_process_lock(entry)

    def _acquire_process_lock(
        self,
        entry: _ProcessLockEntry,
        deadline: float | None,
        cancelled: threading.Event | None,
    ) -> None:
        while True:
            self._raise_if_cancelled(cancelled)
            wait = self._next_wait(deadline)
            if entry.lock.acquire(timeout=wait):
                return
            self._raise_if_timed_out(deadline)

    def _acquire_file_lock(
        self,
        descriptor: int,
        deadline: float | None,
        cancelled: threading.Event | None,
        fcntl: Any,
    ) -> None:
        while True:
            self._raise_if_cancelled(cancelled)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise SessionStorageError(
                        f"failed to acquire Session journal lock {self.path}"
                    ) from exc
            self._raise_if_timed_out(deadline)
            time.sleep(self._next_wait(deadline))

    def _next_wait(self, deadline: float | None) -> float:
        if deadline is None:
            return self.poll_interval
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0
        return min(self.poll_interval, remaining)

    def _raise_if_timed_out(self, deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise SessionLockTimeoutError(
                f"timed out acquiring Session journal lock {self.path}"
            )

    @staticmethod
    def _raise_if_cancelled(cancelled: threading.Event | None) -> None:
        if cancelled is not None and cancelled.is_set():
            raise _SessionLockCancelled

    def _retain_process_lock(self) -> _ProcessLockEntry:
        with _PROCESS_LOCKS_GUARD:
            entry = _PROCESS_LOCKS.get(self._key)
            if entry is None:
                entry = _ProcessLockEntry(lock=threading.Lock())
                _PROCESS_LOCKS[self._key] = entry
            entry.users += 1
            return entry

    def _release_process_lock(self, entry: _ProcessLockEntry) -> None:
        with _PROCESS_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0 and _PROCESS_LOCKS.get(self._key) is entry:
                del _PROCESS_LOCKS[self._key]
