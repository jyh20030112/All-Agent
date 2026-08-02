from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.json import JsonObject, freeze_json_object
from ejagent.contracts.messages import (
    ContextMessage,
    ConversationMessage,
    TransientInstruction,
    is_context_message,
    is_conversation_message,
)
from ejagent.contracts.runs import FailureCode


def _non_negative_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True, slots=True)
class ContextRequest:
    """Immutable inputs for one disposable model ContextView."""

    run_id: str
    source_revision: int
    turn: int
    committed_messages: tuple[ConversationMessage, ...]
    pending_messages: tuple[ConversationMessage, ...] = ()
    transient_instructions: tuple[TransientInstruction, ...] = ()
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("context run_id must not be empty")
        _non_negative_integer(self.source_revision, "context source_revision")
        _non_negative_integer(self.turn, "context turn")
        if self.turn == 0:
            raise ValueError("context turn must be greater than zero")
        object.__setattr__(self, "committed_messages", tuple(self.committed_messages))
        object.__setattr__(self, "pending_messages", tuple(self.pending_messages))
        object.__setattr__(
            self,
            "transient_instructions",
            tuple(self.transient_instructions),
        )
        if not all(
            is_conversation_message(message) for message in self.committed_messages
        ):
            raise TypeError(
                "committed_messages must contain ConversationMessage values"
            )
        if not all(
            is_conversation_message(message) for message in self.pending_messages
        ):
            raise TypeError("pending_messages must contain ConversationMessage values")
        if not all(
            isinstance(instruction, TransientInstruction)
            for instruction in self.transient_instructions
        ):
            raise TypeError(
                "transient_instructions must contain TransientInstruction values"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("context metadata must be a JSON object")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object(self.metadata, label="context metadata"),
        )

    @property
    def messages(self) -> tuple[ContextMessage, ...]:
        return (
            *self.committed_messages,
            *self.pending_messages,
            *self.transient_instructions,
        )


@dataclass(frozen=True, slots=True)
class ContextView:
    """Disposable Provider-neutral projection for one model request."""

    run_id: str
    source_revision: int
    turn: int
    messages: tuple[ContextMessage, ...]
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("ContextView run_id must not be empty")
        _non_negative_integer(self.source_revision, "ContextView source_revision")
        _non_negative_integer(self.turn, "ContextView turn")
        if self.turn == 0:
            raise ValueError("ContextView turn must be greater than zero")
        object.__setattr__(self, "messages", tuple(self.messages))
        if not all(is_context_message(message) for message in self.messages):
            raise TypeError("ContextView messages must contain ContextMessage values")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("ContextView metadata must be a JSON object")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object(self.metadata, label="ContextView metadata"),
        )


@dataclass(frozen=True, slots=True)
class ContextCompactionRequest:
    """Committed history selected for one derived summary projection."""

    messages: tuple[ConversationMessage, ...]
    source_revision_start: int
    source_revision_end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("compaction messages must not be empty")
        if not all(is_conversation_message(message) for message in self.messages):
            raise TypeError(
                "compaction messages must contain ConversationMessage values"
            )
        _non_negative_integer(
            self.source_revision_start,
            "compaction source_revision_start",
        )
        _non_negative_integer(
            self.source_revision_end,
            "compaction source_revision_end",
        )
        if self.source_revision_end < self.source_revision_start:
            raise ValueError("compaction revision range is reversed")


@dataclass(frozen=True, slots=True)
class ContextCompactionOutput:
    """Summary text returned by a ContextCompactor adapter."""

    content: str
    compactor_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("compaction output content must not be empty")
        if not isinstance(self.compactor_id, str) or not self.compactor_id.strip():
            raise ValueError("compaction output compactor_id must not be empty")


class ContextCompactorError(RuntimeError):
    """Expected operational failure from a ContextCompactor."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        if not message.strip():
            raise ValueError("compactor error message must not be empty")
        if not isinstance(retryable, bool):
            raise TypeError("compactor error retryable must be a boolean")
        self.retryable = retryable
        super().__init__(message)


class ContextCompactor(Protocol):
    """Generate summary content without owning Conversation state."""

    async def compact(
        self,
        request: ContextCompactionRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextCompactionOutput:
        """Summarize the selected immutable committed history."""


class ContextBuildError(RuntimeError):
    """Expected operational failure while deriving a ContextView."""

    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        if code not in (
            FailureCode.CONTEXT_OVERFLOW,
            FailureCode.COMPACTION_FAILED,
        ):
            raise ValueError("ContextBuildError requires a context failure code")
        if not message.strip():
            raise ValueError("context error message must not be empty")
        if not isinstance(retryable, bool):
            raise TypeError("context error retryable must be a boolean")
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class ContextProtocolError(RuntimeError):
    """A ContextPipeline violated the stable projection protocol."""


class ContextPipeline(Protocol):
    """Build one ContextView without mutating Conversation or Run state."""

    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        """Return one disposable view for the next model request."""
