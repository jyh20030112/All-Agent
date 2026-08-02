from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar

T = TypeVar("T")


class ControlKind(StrEnum):
    """Kind of input admitted by an AgentHarness control queue."""

    STEERING = "steering"
    FOLLOW_UP = "follow_up"


class ControlStatus(StrEnum):
    """Immediate admission decision for one control input."""

    ACCEPTED = "accepted"
    NOT_RUNNING = "not_running"
    TOO_LATE = "too_late"
    QUEUE_FULL = "queue_full"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ControlReceipt:
    """Stable immediate result of one control admission attempt."""

    input_id: str
    kind: ControlKind
    status: ControlStatus

    def __post_init__(self) -> None:
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise ValueError("control input_id must not be empty")
        if not isinstance(self.kind, ControlKind):
            raise TypeError("control kind must be a ControlKind")
        if not isinstance(self.status, ControlStatus):
            raise TypeError("control status must be a ControlStatus")

    @property
    def accepted(self) -> bool:
        return self.status is ControlStatus.ACCEPTED


@dataclass(frozen=True, slots=True)
class SteeringInput:
    """One admitted transient instruction awaiting a model-call safe point."""

    input_id: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise ValueError("steering input_id must not be empty")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("steering content must not be empty")


class RunControlSource(Protocol):
    """Run-local control source consumed only at Kernel safe points."""

    def drain_steering(self) -> tuple[SteeringInput, ...]:
        """Return admitted steering in FIFO order exactly once."""


class ControlProtocolError(RuntimeError):
    """A RunControlSource violated the stable Kernel control protocol."""


class RunCancelledError(RuntimeError):
    """Raised cooperatively when an active Run is cancelled."""


class CancellationToken:
    """Read-only cancellation signal shared across one Run."""

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
        """Raise the Run-level cancellation exception when cancelled."""

        if self.cancelled:
            raise RunCancelledError(self.reason or "Run was cancelled")

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
        self._reason = reason or "Run was cancelled"
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
