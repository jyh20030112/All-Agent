from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeAlias

from ejagent.contracts.model import ModelUsage

if TYPE_CHECKING:
    from ejagent.agent.cancellation import CancellationToken
    from ejagent.agent.context_builder import ContextBuildResult


class ModelErrorKind(StrEnum):
    """Stable provider-neutral category for model request failures."""

    CONTEXT_OVERFLOW = "context_overflow"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    PROVIDER_ERROR = "provider_error"


class ModelProviderError(RuntimeError):
    """Normalized model provider failure exposed to the Agent Core."""

    kind = ModelErrorKind.PROVIDER_ERROR


class ContextOverflowError(ModelProviderError):
    """The provider rejected a request because its context was too large."""

    kind = ModelErrorKind.CONTEXT_OVERFLOW

    def __init__(self, message: str, *, response_started: bool = False) -> None:
        self.response_started = response_started
        super().__init__(message)


class ModelRateLimitError(ModelProviderError):
    """The provider rejected a request because of rate limiting."""

    kind = ModelErrorKind.RATE_LIMIT


class ModelTimeoutError(ModelProviderError):
    """The provider request exceeded its configured time limit."""

    kind = ModelErrorKind.TIMEOUT


class ModelAuthenticationError(ModelProviderError):
    """The provider rejected the configured credentials."""

    kind = ModelErrorKind.AUTHENTICATION


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """Provider-neutral function call requested by an assistant message."""

    id: str
    name: str
    arguments: str

    def to_agent_message(self) -> dict[str, Any]:
        """Serialize the call into the current conversation message format."""

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """Provider-neutral assistant response consumed by the orchestrator."""

    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()

    def to_agent_message(self) -> dict[str, Any]:
        """Serialize the response for persistent conversation state."""

        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = [
                tool_call.to_agent_message() for tool_call in self.tool_calls
            ]
        return message


def serialize_assistant_message(
    message: AssistantMessage,
    *,
    usage: ModelUsage | None = None,
) -> dict[str, Any]:
    """Attach Core metadata without widening legacy message methods."""

    serialized = dict(message.to_agent_message())
    if usage is not None:
        serialized["usage"] = usage.to_dict()
    return serialized


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    """One provider-neutral piece of assistant text."""

    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            raise ValueError("model text delta must not be empty")


@dataclass(frozen=True, slots=True)
class ModelThinkingDelta:
    """One provider-neutral piece of provisional model reasoning."""

    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            raise ValueError("model thinking delta must not be empty")


@dataclass(frozen=True, slots=True)
class ModelResponseCompleted:
    """Terminal stream event containing the normalized assistant message."""

    message: AssistantMessage
    usage: ModelUsage | None = None


ModelStreamEvent: TypeAlias = (
    ModelTextDelta | ModelThinkingDelta | ModelResponseCompleted
)


class ModelAdapter(ABC):
    """Provider boundary used by the agent core."""

    async def startup(self) -> None:
        """Acquire optional provider resources."""

    async def shutdown(self) -> None:
        """Release optional provider resources."""

    @abstractmethod
    async def complete(
        self,
        context: ContextBuildResult,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        """Return one complete response and honor the per-run cancellation."""

    async def stream(
        self,
        context: ContextBuildResult,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Adapt a complete-only provider into one terminal stream event."""

        message = await self.complete(
            context,
            cancellation=cancellation,
        )
        yield ModelResponseCompleted(message)
