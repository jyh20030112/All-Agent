from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv
from openai import (
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from simagentplg.agent.cancellation import AgentCancelledError
from simagentplg.providers.base import (
    AssistantMessage,
    ContextOverflowError,
    ModelAdapter,
    ModelAuthenticationError,
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

if TYPE_CHECKING:
    from simagentplg.agent.cancellation import CancellationToken
    from simagentplg.agent.context_builder import ContextBuildResult


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Connection and generation settings for an OpenAI-compatible model."""

    model: str
    api_key: str
    base_url: str
    timeout: int = 60
    temperature: float = 0.7
    include_usage: bool = True

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.api_key:
            raise ValueError("api_key must not be empty")
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if not isinstance(self.include_usage, bool):
            raise TypeError("include_usage must be a bool")

    @classmethod
    def from_env(cls) -> ModelConfig:
        """Build a config from the configured model environment variables."""

        load_dotenv()
        model = os.getenv("CHAT_MODEL")
        api_key = os.getenv("MODEL_API_KEY")
        base_url = os.getenv("MODEL_URL")

        if not model or not api_key or not base_url:
            raise ValueError("CHAT_MODEL, MODEL_API_KEY and MODEL_URL must be defined")

        try:
            timeout = int(os.getenv("LLM_TIMEOUT", "60"))
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        except ValueError as exc:
            raise ValueError("LLM_TIMEOUT and LLM_TEMPERATURE must be numeric") from exc

        include_usage_value = os.getenv("LLM_INCLUDE_USAGE", "true").lower()
        if include_usage_value not in {"true", "false"}:
            raise ValueError("LLM_INCLUDE_USAGE must be true or false")

        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            include_usage=include_usage_value == "true",
        )


@dataclass(slots=True)
class _StreamingToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


def _serialize_context_tools(
    context: ContextBuildResult,
) -> tuple[dict[str, Any], ...]:
    """Prefer canonical definitions while preserving legacy contexts."""

    if context.tool_definitions:
        return tuple(tool.to_openai_tool() for tool in context.tool_definitions)
    return context.tools


_CONTEXT_OVERFLOW_CODES = {
    "context_length_error",
    "context_length_exceeded",
    "context_window_exceeded",
    "input_too_long",
    "prompt_too_long",
}
_CONTEXT_OVERFLOW_PHRASES = (
    "context length exceeded",
    "context window exceeded",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
)


def _provider_error_values(exc: Exception) -> tuple[str, ...]:
    values: list[str] = []
    for candidate in (
        getattr(exc, "code", None),
        getattr(exc, "type", None),
        getattr(exc, "body", None),
    ):
        if isinstance(candidate, str):
            values.append(candidate)
        elif isinstance(candidate, Mapping):
            for key in ("code", "type", "message"):
                value = candidate.get(key)
                if isinstance(value, str):
                    values.append(value)
            nested = candidate.get("error")
            if isinstance(nested, Mapping):
                for key in ("code", "type", "message"):
                    value = nested.get(key)
                    if isinstance(value, str):
                        values.append(value)
    values.append(str(exc))
    return tuple(values)


def _is_context_overflow(exc: Exception) -> bool:
    values = _provider_error_values(exc)
    normalized_codes = {value.strip().lower() for value in values}
    if normalized_codes & _CONTEXT_OVERFLOW_CODES:
        return True
    text = " ".join(normalized_codes)
    return any(phrase in text for phrase in _CONTEXT_OVERFLOW_PHRASES)


def _normalize_provider_error(
    exc: Exception,
    *,
    operation: str,
) -> ModelProviderError:
    if isinstance(exc, ModelProviderError):
        return exc
    message = f"{operation} failed: {exc}"
    if isinstance(exc, AuthenticationError):
        return ModelAuthenticationError(message)
    if isinstance(exc, RateLimitError):
        return ModelRateLimitError(message)
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return ModelTimeoutError(message)
    if _is_context_overflow(exc):
        return ContextOverflowError(message)
    return ModelProviderError(message)


def _usage_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_usage(raw_usage: Any) -> ModelUsage:
    input_tokens = int(_usage_field(raw_usage, "prompt_tokens") or 0)
    output_tokens = int(_usage_field(raw_usage, "completion_tokens") or 0)
    prompt_details = _usage_field(raw_usage, "prompt_tokens_details")
    completion_details = _usage_field(
        raw_usage,
        "completion_tokens_details",
    )
    cache_read = _usage_field(prompt_details, "cached_tokens")
    cache_write = _usage_field(prompt_details, "cache_write_tokens")
    reasoning = _usage_field(completion_details, "reasoning_tokens")
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=int(cache_read) if cache_read is not None else None,
        cache_write_tokens=(int(cache_write) if cache_write is not None else None),
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
    )


class OpenAIModelAdapter(ModelAdapter):
    """OpenAI-compatible provider adapter for the core model contract."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._owns_client = client is None

    async def startup(self) -> None:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )

    async def shutdown(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None

    async def complete(
        self,
        context: ContextBuildResult,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        await self.startup()
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI model client is not initialized")

        tools = _serialize_context_tools(context)
        try:
            request = client.chat.completions.create(
                model=self.config.model,
                messages=cast(Any, context.llm_messages),
                temperature=self.config.temperature,
                tools=cast(Any, tools or None),
            )
            response = (
                await cancellation.run(request)
                if cancellation is not None
                else await request
            )
        except asyncio.CancelledError:
            raise
        except AgentCancelledError:
            raise
        except Exception as exc:
            raise _normalize_provider_error(
                exc,
                operation="chat completion",
            ) from exc

        if not response.choices:
            raise ModelProviderError("chat completion returned no choices")
        message = response.choices[0].message
        return AssistantMessage(
            content=message.content,
            tool_calls=tuple(
                ModelToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in message.tool_calls or ()
                if tool_call.type == "function"
            ),
        )

    async def stream(
        self,
        context: ContextBuildResult,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream and normalize one OpenAI-compatible chat completion."""

        await self.startup()
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI model client is not initialized")

        tools = _serialize_context_tools(context)
        response: Any | None = None
        content_parts: list[str] = []
        tool_calls: dict[int, _StreamingToolCall] = {}
        has_finish_reason = False
        usage: ModelUsage | None = None
        try:
            request_options: dict[str, Any] = {
                "model": self.config.model,
                "messages": cast(Any, context.llm_messages),
                "temperature": self.config.temperature,
                "tools": cast(Any, tools) or None,
                "stream": True,
            }
            if self.config.include_usage:
                request_options["stream_options"] = {"include_usage": True}
            request = client.chat.completions.create(**request_options)
            response = (
                await cancellation.run(request)
                if cancellation is not None
                else await request
            )
            assert response is not None
            iterator = response.__aiter__()
            while True:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                try:
                    next_chunk = anext(iterator)
                    chunk = (
                        await cancellation.run(next_chunk)
                        if cancellation is not None
                        else await next_chunk
                    )
                except StopAsyncIteration:
                    break

                raw_usage = getattr(chunk, "usage", None)
                if raw_usage is not None:
                    usage = _normalize_usage(raw_usage)

                choices = getattr(chunk, "choices", None) or ()
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None) is not None:
                    has_finish_reason = True
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if content:
                    text = str(content)
                    content_parts.append(text)
                    yield ModelTextDelta(text)

                for field in (
                    "reasoning_content",
                    "reasoning",
                    "reasoning_text",
                ):
                    reasoning = getattr(delta, field, None)
                    if isinstance(reasoning, str) and reasoning:
                        yield ModelThinkingDelta(reasoning)
                        break

                for position, partial in enumerate(
                    getattr(delta, "tool_calls", None) or ()
                ):
                    index = getattr(partial, "index", None)
                    if index is None:
                        index = position
                    call = tool_calls.setdefault(index, _StreamingToolCall())
                    call_id = getattr(partial, "id", None)
                    if call_id:
                        call.id = str(call_id)
                    function = getattr(partial, "function", None)
                    if function is None:
                        continue
                    name = getattr(function, "name", None)
                    if name:
                        call.name += str(name)
                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        call.arguments += str(arguments)
        except asyncio.CancelledError:
            raise
        except AgentCancelledError:
            raise
        except Exception as exc:
            raise _normalize_provider_error(
                exc,
                operation="chat completion stream",
            ) from exc
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if close is None:
                    close = getattr(response, "aclose", None)
                if close is not None:
                    with suppress(Exception):
                        close_result = close()
                        if inspect.isawaitable(close_result):
                            await close_result

        if not has_finish_reason:
            raise ModelProviderError(
                "chat completion stream ended without finish_reason"
            )

        normalized_tool_calls: list[ModelToolCall] = []
        for index in sorted(tool_calls):
            call = tool_calls[index]
            if not call.id or not call.name:
                raise ModelProviderError(
                    "chat completion stream returned an incomplete tool call"
                )
            normalized_tool_calls.append(
                ModelToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                )
            )

        yield ModelResponseCompleted(
            AssistantMessage(
                content="".join(content_parts) or None,
                tool_calls=tuple(normalized_tool_calls),
            ),
            usage=usage,
        )
