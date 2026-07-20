from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from simagentplg.agent.result import AgentRunResult


class ControlInputKind(StrEnum):
    """Kind of external input submitted to an active Agent operation."""

    STEERING = "steering"
    FOLLOW_UP = "follow_up"


class ControlStatus(StrEnum):
    """Immediate result of submitting one control input."""

    ACCEPTED = "accepted"
    AGENT_IDLE = "agent_idle"
    QUEUE_FULL = "queue_full"
    RUN_CLOSING = "run_closing"


class FollowUpFailurePolicy(StrEnum):
    """Whether a Run chain continues after a non-successful result."""

    DISCARD = "discard"
    CONTINUE = "continue"


class FollowUpDiscardReason(StrEnum):
    """Reason an accepted Follow-up will not start a Run."""

    PREVIOUS_RUN_NOT_COMPLETED = "previous_run_not_completed"
    AGENT_SHUTDOWN = "agent_shutdown"


@dataclass(frozen=True, slots=True)
class ControlInput:
    """One immutable external instruction with a stable correlation id."""

    input_id: str
    kind: ControlInputKind
    content: str

    def __post_init__(self) -> None:
        input_id = self.input_id.strip()
        content = self.content.strip()
        if not input_id:
            raise ValueError("input_id must not be empty")
        if not content:
            raise ValueError("control content must not be empty")
        object.__setattr__(self, "input_id", input_id)
        object.__setattr__(self, "content", content)

    @classmethod
    def steering(cls, content: str) -> ControlInput:
        """Create one Steering instruction for the active Run."""

        return cls(
            input_id=uuid4().hex,
            kind=ControlInputKind.STEERING,
            content=content,
        )

    @classmethod
    def follow_up(cls, task: str) -> ControlInput:
        """Create one Follow-up task for the active Run chain."""

        return cls(
            input_id=uuid4().hex,
            kind=ControlInputKind.FOLLOW_UP,
            content=task,
        )


@dataclass(frozen=True, slots=True)
class ControlReceipt:
    """Immediate acknowledgement for one submitted control input."""

    control: ControlInput
    status: ControlStatus
    queue_size: int

    def __post_init__(self) -> None:
        if self.queue_size < 0:
            raise ValueError("queue_size must not be negative")

    @property
    def accepted(self) -> bool:
        """Return whether the input entered the active Run's queue."""

        return self.status is ControlStatus.ACCEPTED


class FollowUpError(RuntimeError):
    """Base error raised while waiting for a Follow-up Run."""


class FollowUpRejectedError(FollowUpError):
    """The Follow-up never entered the Run chain."""

    def __init__(self, receipt: ControlReceipt) -> None:
        self.receipt = receipt
        super().__init__(f"follow-up was not accepted: {receipt.status.value}")


class FollowUpDiscardedError(FollowUpError):
    """An accepted Follow-up was discarded before its Run started."""

    def __init__(
        self,
        control: ControlInput,
        reason: FollowUpDiscardReason,
    ) -> None:
        self.control = control
        self.reason = reason
        super().__init__(f"follow-up was discarded: {reason.value}")


class FollowUpHandle:
    """Waitable handle for one accepted or rejected Follow-up submission."""

    def __init__(self, receipt: ControlReceipt) -> None:
        if receipt.control.kind is not ControlInputKind.FOLLOW_UP:
            raise ValueError("FollowUpHandle requires a Follow-up control input")
        self.receipt = receipt
        self._future: asyncio.Future[AgentRunResult] = (
            asyncio.get_running_loop().create_future()
        )
        # Retrieving the exception here prevents warnings when a rejected or
        # discarded handle is intentionally never awaited. Awaiters still see it.
        self._future.add_done_callback(self._consume_exception)
        if not receipt.accepted:
            self._future.set_exception(FollowUpRejectedError(receipt))

    @staticmethod
    def _consume_exception(future: asyncio.Future[AgentRunResult]) -> None:
        if not future.cancelled():
            future.exception()

    @property
    def control(self) -> ControlInput:
        """Return the immutable task and correlation id."""

        return self.receipt.control

    @property
    def accepted(self) -> bool:
        """Return whether this Follow-up entered the pending queue."""

        return self.receipt.accepted

    @property
    def done(self) -> bool:
        """Return whether the Run completed or the submission was rejected."""

        return self._future.done()

    async def wait(self) -> AgentRunResult:
        """Wait without allowing caller cancellation to cancel the queued Run."""

        return await asyncio.shield(self._future)

    def _set_result(self, result: AgentRunResult) -> None:
        if not self._future.done():
            self._future.set_result(result)

    def _set_exception(self, error: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(error)

    def _discard(self, reason: FollowUpDiscardReason) -> None:
        self._set_exception(FollowUpDiscardedError(self.control, reason))


class _SteeringQueue:
    """Run-scoped bounded FIFO consumed only at model-call safe points."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: deque[ControlInput] = deque()
        self._closed = False

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def has_pending(self) -> bool:
        return bool(self._items)

    def submit(self, control: ControlInput) -> ControlReceipt:
        if self._closed:
            status = ControlStatus.RUN_CLOSING
        elif len(self._items) >= self.capacity:
            status = ControlStatus.QUEUE_FULL
        else:
            self._items.append(control)
            status = ControlStatus.ACCEPTED
        return ControlReceipt(
            control=control,
            status=status,
            queue_size=len(self._items),
        )

    def drain(self) -> tuple[ControlInput, ...]:
        items = tuple(self._items)
        self._items.clear()
        return items

    def close(self) -> tuple[ControlInput, ...]:
        self._closed = True
        return self.drain()


class _FollowUpQueue:
    """Agent-scoped bounded FIFO of independently waitable tasks."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: deque[FollowUpHandle] = deque()

    @property
    def size(self) -> int:
        return len(self._items)

    def submit(self, control: ControlInput) -> FollowUpHandle:
        if len(self._items) >= self.capacity:
            return self.rejected(control, ControlStatus.QUEUE_FULL)
        handle = FollowUpHandle(
            ControlReceipt(
                control=control,
                status=ControlStatus.ACCEPTED,
                queue_size=len(self._items) + 1,
            )
        )
        self._items.append(handle)
        return handle

    def rejected(
        self,
        control: ControlInput,
        status: ControlStatus,
    ) -> FollowUpHandle:
        return FollowUpHandle(
            ControlReceipt(
                control=control,
                status=status,
                queue_size=len(self._items),
            )
        )

    def pop(self) -> FollowUpHandle | None:
        if not self._items:
            return None
        return self._items.popleft()

    def drain(self) -> tuple[FollowUpHandle, ...]:
        items = tuple(self._items)
        self._items.clear()
        return items
