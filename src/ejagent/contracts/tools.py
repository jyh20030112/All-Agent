from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.json import (
    JsonObject,
    JsonValue,
    freeze_json_object,
    freeze_json_value,
)
from ejagent.contracts.messages import ToolCall

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ToolControl(StrEnum):
    """Terminal control requested by one Tool execution."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    REJECT = "reject"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Provider-neutral Tool definition exposed to a Runtime Kernel."""

    name: str
    description: str | None = None
    input_schema: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _TOOL_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ValueError(
                "tool name must contain 1-64 letters, digits, underscores, or dashes"
            )
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("tool description must be a string or None")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("tool input_schema must be a JSON object")
        object.__setattr__(
            self,
            "input_schema",
            freeze_json_object(self.input_schema, label="tool input_schema"),
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Normalized operational result returned by a ToolExecutor."""

    result: JsonValue
    control: ToolControl = ToolControl.CONTINUE
    output: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result",
            freeze_json_value(self.result, label="tool execution result"),
        )
        if not isinstance(self.control, ToolControl):
            raise TypeError("tool control must be a ToolControl")
        if self.output is not None and not isinstance(self.output, str):
            raise TypeError("tool output must be a string or None")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise ValueError("tool error must not be empty")

    @property
    def is_error(self) -> bool:
        return self.error is not None


class ToolExecutionError(RuntimeError):
    """Expected Tool infrastructure failure that prevents a result."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        if not message.strip():
            raise ValueError("Tool execution error message must not be empty")
        if not isinstance(retryable, bool):
            raise TypeError("Tool execution retryable must be a boolean")
        self.retryable = retryable
        super().__init__(message)


class ToolProtocolError(RuntimeError):
    """A ToolExecutor violated the stable Kernel protocol."""


class ToolExecutor(Protocol):
    """Ready Tool execution boundary consumed by a Runtime Kernel."""

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        """Return the immutable Tool collection captured for a Run."""

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        """Execute one Tool call without acquiring or releasing resources."""
