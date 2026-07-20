from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4


class ControlInputKind(StrEnum):
    """Kind of external input submitted to an active Agent operation."""

    STEERING = "steering"


class ControlStatus(StrEnum):
    """Immediate result of submitting one control input."""

    ACCEPTED = "accepted"
    AGENT_IDLE = "agent_idle"
    QUEUE_FULL = "queue_full"
    RUN_CLOSING = "run_closing"


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
