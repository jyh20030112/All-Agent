from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ejagent.contracts.runs import AuditRecord, RunFailure, RunResult


@dataclass(frozen=True, slots=True)
class RunAudit:
    """Append-only facts for one attempted Run, separate from Conversation."""

    result: RunResult
    base_revision: int
    resulting_revision: int
    committed: bool
    records: tuple[AuditRecord, ...] = ()
    failure: RunFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, RunResult):
            raise TypeError("audit result must be a RunResult")
        for name, value in (
            ("base_revision", self.base_revision),
            ("resulting_revision", self.resulting_revision),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"audit {name} must be an integer")
            if value < 0:
                raise ValueError(f"audit {name} must not be negative")
        if not isinstance(self.committed, bool):
            raise TypeError("audit committed must be a boolean")
        expected_revision = self.base_revision + int(self.committed)
        if self.resulting_revision != expected_revision:
            raise ValueError("audit resulting revision does not match commit status")
        object.__setattr__(self, "records", tuple(self.records))
        if not all(isinstance(record, AuditRecord) for record in self.records):
            raise TypeError("audit records must contain AuditRecord values")
        if any(record.run_id != self.result.run_id for record in self.records):
            raise ValueError("audit records must belong to the audited Run")
        if self.failure is not None and not isinstance(self.failure, RunFailure):
            raise TypeError("audit failure must be a RunFailure or None")

    @property
    def run_id(self) -> str:
        return self.result.run_id


class AuditReader(Protocol):
    """Read-only seam for durable Run audit history."""

    async def load_audit(self, agent_id: str) -> tuple[RunAudit, ...]:
        """Return Run facts in durable insertion order."""
