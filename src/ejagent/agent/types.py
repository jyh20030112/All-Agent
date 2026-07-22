from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

AgentMessage = dict[str, Any]
INTERNAL_METADATA_PREFIX = "_ejagent_"
LEGACY_INTERNAL_METADATA_PREFIXES = ("_simagentplg_",)
INTERNAL_METADATA_PREFIXES = (
    INTERNAL_METADATA_PREFIX,
    *LEGACY_INTERNAL_METADATA_PREFIXES,
)


def is_internal_metadata_key(key: str) -> bool:
    """Return whether a message key belongs to current or legacy Core metadata."""

    return key.startswith(INTERNAL_METADATA_PREFIXES)


class ToolControl(StrEnum):
    """Control signal returned by a tool independently of its payload."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    REJECT = "reject"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class ToolProgressUpdate:
    """One provisional, human-readable update from an executing tool."""

    message: str
    data: Any = None

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("tool progress message must not be empty")


class ToolProgressReporter(Protocol):
    """Scoped interface used by one tool call to publish progress."""

    async def report(self, update: ToolProgressUpdate) -> None:
        """Publish an update while the owning tool call is active."""


@dataclass(slots=True)
class StepOutcome:
    """Normalized result returned by every tool handler."""

    data: Any
    control: ToolControl = ToolControl.CONTINUE


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Normalized result of one model-requested tool execution."""

    messages: tuple[AgentMessage, ...]
    control: ToolControl = ToolControl.CONTINUE
    output: str | None = None
    error: str | None = None
    cancelled: bool = False
