"""Model provider adapters for the agent core."""

from ejagent.providers.base import (
    AssistantMessage,
    ContextOverflowError,
    ModelAdapter,
    ModelAuthenticationError,
    ModelErrorKind,
    ModelProviderError,
    ModelRateLimitError,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelTimeoutError,
    ModelToolCall,
    ModelUsage,
)
from ejagent.providers.openai import ModelConfig, OpenAIModelAdapter

__all__ = [
    "AssistantMessage",
    "ModelErrorKind",
    "ModelProviderError",
    "ContextOverflowError",
    "ModelRateLimitError",
    "ModelTimeoutError",
    "ModelAuthenticationError",
    "ModelAdapter",
    "ModelStreamEvent",
    "ModelTextDelta",
    "ModelThinkingDelta",
    "ModelResponseCompleted",
    "ModelToolCall",
    "ModelUsage",
    "ModelConfig",
    "OpenAIModelAdapter",
]
