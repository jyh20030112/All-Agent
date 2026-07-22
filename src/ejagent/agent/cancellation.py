from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


class AgentCancelledError(RuntimeError):
    """Raised cooperatively when the active agent run is aborted."""


class CancellationToken:
    """Read-only cancellation signal shared across one agent run."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """Return the cancellation reason when one was supplied."""

        return self._reason

    async def wait(self) -> None:
        """Wait until cancellation is requested."""

        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        """Raise the agent-level cancellation exception when cancelled."""

        if self.cancelled:
            raise AgentCancelledError(self.reason or "agent run was aborted")

    async def run(self, awaitable: Awaitable[T]) -> T:
        """Await work while interrupting it when this token is cancelled."""

        if self.cancelled:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            self.raise_if_cancelled()

        work = asyncio.ensure_future(awaitable)
        cancellation = asyncio.create_task(self.wait())
        try:
            done, _ = await asyncio.wait(
                (work, cancellation),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if work in done:
                return await work

            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            self.raise_if_cancelled()
            raise RuntimeError("cancellation wait completed without a signal")
        finally:
            cancellation.cancel()
            if not work.done():
                work.cancel()
            await asyncio.gather(
                work,
                cancellation,
                return_exceptions=True,
            )

    def _cancel(self, reason: str | None) -> bool:
        if self.cancelled:
            return False
        self._reason = reason or "agent run was aborted"
        self._event.set()
        return True


class CancellationSource:
    """Mutable owner of one public read-only cancellation token."""

    def __init__(self) -> None:
        self._token = CancellationToken()

    @property
    def token(self) -> CancellationToken:
        return self._token

    def cancel(self, reason: str | None = None) -> bool:
        """Request cancellation once and report whether state changed."""

        return self._token._cancel(reason)
