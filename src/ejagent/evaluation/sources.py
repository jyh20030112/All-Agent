"""Bounded file observations and Run-scoped tool concurrency probes."""

from __future__ import annotations

import asyncio
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.json import JsonObject
from ejagent.evaluation.types import EvidenceSnapshot, EvidenceUnavailable, fingerprint
from ejagent.kernel.trajectory import CheckpointSignal


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class FileEvidenceSource:
    """Read one explicitly configured UTF-8 artifact, with a bounded byte size."""

    def __init__(self, path: str | Path, *, max_bytes: int = 262_144) -> None:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        self._path = Path(path).absolute()
        self._max_bytes = max_bytes

    def _snapshot(self) -> EvidenceSnapshot:
        try:
            descriptor = os.open(self._path, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                # Stat both sides of the read catches concurrent replacement/writes.
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise EvidenceUnavailable("artifact is not a regular file")
                content = stream.read(self._max_bytes + 1)
                after = self._path.stat()
            if _stat_signature(before) != _stat_signature(after):
                raise EvidenceUnavailable("file changed during read")
            if len(content) > self._max_bytes:
                raise EvidenceUnavailable("file exceeds configured size bound")
            text = content.decode("utf-8")
            value: JsonObject = {"exists": True, "text": text}
            revision = fingerprint(
                {"stat": _stat_signature(after), "content": fingerprint(text)}
            )
        except FileNotFoundError:
            # Absence is observed evidence, not a failed attempt to read content.
            value = {"exists": False, "text": None}
            revision = "absent"
        except UnicodeDecodeError as exc:
            raise EvidenceUnavailable("file is not valid UTF-8") from exc
        return EvidenceSnapshot(revision, value, str(self._path))

    async def revision(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> str:
        cancellation.raise_if_cancelled()
        return (await cancellation.run(asyncio.to_thread(self._snapshot))).revision

    async def read(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> EvidenceSnapshot:
        cancellation.raise_if_cancelled()
        return await cancellation.run(asyncio.to_thread(self._snapshot))

    def close_run(self, run_id: str) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _Probe:
    name: str
    start: float
    end: float


class ProbeEvidenceSource:
    """Record completed probe intervals, exposing stable goal-relevant facts."""

    def __init__(
        self,
        names: tuple[str, str] = ("probe_a", "probe_b"),
        *,
        max_records: int = 1024,
    ) -> None:
        if len(names) != 2 or len(set(names)) != 2 or any(not name for name in names):
            raise ValueError("two distinct probe names are required")
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or max_records <= 0
        ):
            raise ValueError("max_records must be a positive integer")
        self._names = tuple(names)
        self._max_records = max_records
        self._records: dict[str, list[_Probe]] = {}
        self._overflow: set[str] = set()

    def record(
        self,
        run_id: str,
        tool_name: str,
        *,
        started_at: float,
        finished_at: float,
        cancelled: bool = False,
    ) -> None:
        if not run_id or tool_name not in self._names:
            raise ValueError("record requires a Run ID and configured probe name")
        if (
            not all(math.isfinite(value) for value in (started_at, finished_at))
            or finished_at < started_at
        ):
            raise ValueError("probe interval must be finite and ordered")
        if cancelled:
            return
        records = self._records.setdefault(run_id, [])
        if len(records) >= self._max_records:
            self._overflow.add(run_id)
        else:
            records.append(_Probe(tool_name, started_at, finished_at))

    def _snapshot(self, run_id: str) -> EvidenceSnapshot:
        if run_id in self._overflow:
            raise EvidenceUnavailable("probe record limit exceeded")
        records = self._records.get(run_id, [])
        first = [item for item in records if item.name == self._names[0]]
        second = [item for item in records if item.name == self._names[1]]
        value: JsonObject = {
            self._names[0]: bool(first),
            self._names[1]: bool(second),
            "overlapped": any(
                max(a.start, b.start) < min(a.end, b.end) for a in first for b in second
            ),
        }
        # Additional equivalent intervals do not change the semantic observation.
        return EvidenceSnapshot(str(len(records)), value, f"probe-recorder:{run_id}")

    async def revision(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> str:
        cancellation.raise_if_cancelled()
        return self._snapshot(signal.run_id).revision

    async def read(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> EvidenceSnapshot:
        cancellation.raise_if_cancelled()
        return self._snapshot(signal.run_id)

    def close_run(self, run_id: str) -> None:
        self._records.pop(run_id, None)
        self._overflow.discard(run_id)
