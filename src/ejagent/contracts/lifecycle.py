from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ManagedResource(Protocol):
    """Resource whose lifetime is owned by one AgentHarness."""

    async def start(self) -> None:
        """Acquire resources before the Harness accepts Runs."""

    async def shutdown(self) -> None:
        """Release resources after the Harness stops accepting Runs."""
