from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from ejagent.contracts.control import ControlReceipt, SteeringInput
from ejagent.contracts.runs import RunOutcome


class FollowUpRejectedError(RuntimeError):
    """A follow-up was not admitted to an active Run chain."""

    def __init__(self, receipt: ControlReceipt) -> None:
        self.receipt = receipt
        super().__init__(f"follow-up was not accepted: {receipt.status.value}")


class FollowUpDiscardedError(RuntimeError):
    """An accepted follow-up was discarded before its Run started."""


class FollowUpHandle:
    """Asynchronous result holder returned immediately by follow_up()."""

    def __init__(self, receipt: ControlReceipt) -> None:
        self.receipt = receipt
        self._event = asyncio.Event()
        self._outcome: RunOutcome | None = None
        self._error: BaseException | None = None

    @property
    def accepted(self) -> bool:
        return self.receipt.accepted

    async def wait(self) -> RunOutcome:
        """Wait for the admitted Run or raise its admission/execution error."""

        if not self.accepted:
            raise FollowUpRejectedError(self.receipt)
        await self._event.wait()
        if self._error is not None:
            raise self._error
        assert self._outcome is not None
        return self._outcome

    def _resolve(self, outcome: RunOutcome) -> None:
        if self._event.is_set():
            return
        self._outcome = outcome
        self._event.set()

    def _fail(self, error: BaseException) -> None:
        if self._event.is_set():
            return
        self._error = error
        self._event.set()


@dataclass(frozen=True, slots=True)
class _QueuedFollowUp:
    task: str
    handle: FollowUpHandle


class _RunControls:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._steering: deque[SteeringInput] = deque()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def offer(self, item: SteeringInput) -> bool:
        if self._closed or len(self._steering) >= self._capacity:
            return False
        self._steering.append(item)
        return True

    def drain_steering(self) -> tuple[SteeringInput, ...]:
        items = tuple(self._steering)
        self._steering.clear()
        return items

    def close(self) -> tuple[SteeringInput, ...]:
        self._closed = True
        return self.drain_steering()
