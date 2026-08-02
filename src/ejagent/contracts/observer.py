from __future__ import annotations

from typing import Protocol

from ejagent.contracts.audit import RunAudit


class RunObserver(Protocol):
    """Observe a finished Run without participating in its result or commit."""

    async def observe(self, audit: RunAudit) -> None:
        """Consume one immutable RunAudit after the Store decision."""
