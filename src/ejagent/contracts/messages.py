from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias, TypeGuard

from ejagent.contracts.json import (
    JsonObject,
    JsonValue,
    freeze_json_object,
    freeze_json_value,
)

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Provider-neutral Tool invocation requested by an assistant."""

    id: str
    name: str
    arguments: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.id, "tool call id")
        if not isinstance(self.name, str) or not _TOOL_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ValueError(
                "tool call name must contain 1-64 letters, digits, "
                "underscores, or dashes"
            )
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be a JSON object")
        object.__setattr__(
            self,
            "arguments",
            freeze_json_object(self.arguments, label="tool call arguments"),
        )


@dataclass(frozen=True, slots=True)
class SystemMessage:
    """Stable instruction belonging to committed conversation state."""

    content: str

    def __post_init__(self) -> None:
        _required_text(self.content, "system message content")


@dataclass(frozen=True, slots=True)
class UserMessage:
    """One user input committed to conversation state."""

    content: str

    def __post_init__(self) -> None:
        _required_text(self.content, "user message content")


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """Provider-neutral assistant output committed as one message."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("assistant message content must be a string or None")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if not self.content and not self.tool_calls:
            raise ValueError("assistant message must contain text or tool calls")
        if not all(isinstance(call, ToolCall) for call in self.tool_calls):
            raise TypeError("assistant tool_calls must contain ToolCall values")


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """Provider-neutral result paired with one committed Tool call."""

    tool_call_id: str
    tool_name: str
    result: JsonValue
    is_error: bool = False

    def __post_init__(self) -> None:
        _required_text(self.tool_call_id, "tool result call id")
        if not isinstance(self.tool_name, str) or not _TOOL_NAME_PATTERN.fullmatch(
            self.tool_name
        ):
            raise ValueError(
                "tool result name must contain 1-64 letters, digits, "
                "underscores, or dashes"
            )
        if not isinstance(self.is_error, bool):
            raise TypeError("tool result is_error must be a boolean")
        object.__setattr__(
            self,
            "result",
            freeze_json_value(self.result, label="tool result"),
        )


@dataclass(frozen=True, slots=True)
class ContextSummary:
    """Derived summary usable in a ContextView but not conversation truth."""

    source_revision_start: int
    source_revision_end: int
    content: str
    compactor_id: str

    def __post_init__(self) -> None:
        if self.source_revision_start < 0:
            raise ValueError("summary start revision must not be negative")
        if self.source_revision_end < self.source_revision_start:
            raise ValueError("summary end revision must not precede start revision")
        _required_text(self.content, "summary content")
        _required_text(self.compactor_id, "summary compactor_id")


ConversationMessage: TypeAlias = (
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage
)
ContextMessage: TypeAlias = ConversationMessage | ContextSummary


def is_conversation_message(value: object) -> TypeGuard[ConversationMessage]:
    """Return whether a value belongs to the closed Conversation union."""

    return isinstance(
        value,
        (SystemMessage, UserMessage, AssistantMessage, ToolResultMessage),
    )
