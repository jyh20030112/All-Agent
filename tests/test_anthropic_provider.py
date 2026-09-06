from __future__ import annotations

import unittest
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    ContextSummary,
    FailureCode,
    ModelCallError,
    ModelProtocolError,
    ModelRequest,
    ModelResponseCompleted,
    ModelTextDelta,
    ModelThinkingDelta,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    TransientInstruction,
    UserMessage,
)
from ejagent.providers import (
    AnthropicConfig,
    AnthropicModelPort,
    ModelConfig,
    OpenAIModelPort,
)


class FakeAnthropicStream:
    def __init__(self, events: Sequence[Any]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> FakeAnthropicStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeAnthropicManager:
    def __init__(self, result: Sequence[Any] | Exception) -> None:
        self.result = result
        self.exited = False

    async def __aenter__(self) -> FakeAnthropicStream:
        if isinstance(self.result, Exception):
            raise self.result
        return FakeAnthropicStream(self.result)

    async def __aexit__(self, *_: Any) -> None:
        self.exited = True


class FakeAnthropicMessages:
    def __init__(self, results: Sequence[Sequence[Any] | Exception]) -> None:
        self.results = list(results)
        self.requests: list[dict[str, Any]] = []
        self.managers: list[FakeAnthropicManager] = []

    def stream(self, **kwargs: Any) -> FakeAnthropicManager:
        self.requests.append(kwargs)
        manager = FakeAnthropicManager(self.results.pop(0))
        self.managers.append(manager)
        return manager


class FakeAnthropicClient:
    def __init__(self, messages: FakeAnthropicMessages) -> None:
        self.messages = messages


def _anthropic_config() -> AnthropicConfig:
    return AnthropicConfig(
        model="test-claude",
        api_key="test-key",
        max_tokens=512,
        base_url="https://anthropic.invalid",
    )


def _text_events(text: str = "ok") -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 2, "output_tokens": 1}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


class AnthropicModelPortTests(unittest.IsolatedAsyncioTestCase):
    async def test_serializes_content_blocks_and_normalizes_stream(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 2,
                        "cache_creation_input_tokens": 1,
                    }
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "想"},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "答"},
            },
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu-2",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"q":',
                },
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '"值"}',
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
        messages = FakeAnthropicMessages([events])
        port = AnthropicModelPort(
            _anthropic_config(),
            client=FakeAnthropicClient(messages),
        )
        await port.start()
        request = ModelRequest(
            max_output_tokens=128,
            messages=(
                SystemMessage("stable"),
                UserMessage("hello"),
                AssistantMessage(
                    tool_calls=(
                        ToolCall("toolu-0", "lookup", {"q": "first"}),
                        ToolCall("toolu-1", "lookup", {"q": "second"}),
                    )
                ),
                ToolResultMessage("toolu-0", "lookup", {"found": True}),
                ToolResultMessage("toolu-1", "lookup", "missing", is_error=True),
                ContextSummary(1, 3, "summary", "compact-v1"),
                TransientInstruction("focus", "steering"),
            ),
            tools=(
                ToolDefinition(
                    "lookup",
                    description="Lookup a value.",
                    input_schema={"type": "object"},
                ),
            ),
        )

        # The request cap is lower than the model configuration cap (512).
        normalized = [
            event
            async for event in port.stream(
                request,
                cancellation=CancellationSource().token,
            )
        ]

        self.assertEqual(messages.requests[0]["max_tokens"], 128)
        self.assertEqual(normalized[0], ModelThinkingDelta("想"))
        self.assertEqual(normalized[1], ModelTextDelta("答"))
        completed = normalized[2]
        self.assertIsInstance(completed, ModelResponseCompleted)
        assert isinstance(completed, ModelResponseCompleted)
        self.assertEqual(
            completed.message,
            AssistantMessage(
                "答",
                (ToolCall("toolu-2", "lookup", {"q": "值"}),),
            ),
        )
        assert completed.usage is not None
        self.assertEqual(completed.usage.input_tokens, 10)
        self.assertEqual(completed.usage.output_tokens, 5)
        self.assertEqual(completed.usage.cache_read_tokens, 2)
        self.assertEqual(completed.usage.cache_write_tokens, 1)

        sent = messages.requests[0]
        self.assertEqual(sent["system"].split("\n\n")[0], "stable")
        self.assertIn("Derived summary", sent["system"])
        self.assertIn("Transient instruction", sent["system"])
        self.assertEqual(sent["messages"][0]["role"], "user")
        assistant_blocks = sent["messages"][1]["content"]
        self.assertEqual(assistant_blocks[0]["type"], "tool_use")
        self.assertEqual(assistant_blocks[0]["input"], {"q": "first"})
        result_blocks = sent["messages"][2]["content"]
        self.assertEqual(len(result_blocks), 2)
        self.assertEqual(result_blocks[0]["content"], '{"found":true}')
        self.assertTrue(result_blocks[1]["is_error"])
        self.assertEqual(sent["tools"][0]["input_schema"], {"type": "object"})
        self.assertTrue(messages.managers[0].exited)

    async def test_rejects_incomplete_tool_input_as_protocol_error(self) -> None:
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu-0",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "not-json",
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
            },
            {"type": "message_stop"},
        ]
        port = AnthropicModelPort(
            _anthropic_config(),
            client=FakeAnthropicClient(FakeAnthropicMessages([events])),
        )
        await port.start()

        with self.assertRaises(ModelProtocolError):
            async for _ in port.stream(
                ModelRequest(messages=(UserMessage("lookup"),)),
                cancellation=CancellationSource().token,
            ):
                pass

    async def test_normalizes_provider_timeout(self) -> None:
        port = AnthropicModelPort(
            _anthropic_config(),
            client=FakeAnthropicClient(FakeAnthropicMessages([TimeoutError("late")])),
        )
        await port.start()

        with self.assertRaises(ModelCallError) as raised:
            async for _ in port.stream(
                ModelRequest(messages=(UserMessage("hello"),)),
                cancellation=CancellationSource().token,
            ):
                pass

        self.assertEqual(raised.exception.code, FailureCode.TIMEOUT)
        self.assertTrue(raised.exception.retryable)


class _FakeOpenAIStream:
    def __init__(self) -> None:
        self._chunks = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(
                                content="ok",
                                reasoning_content=None,
                                tool_calls=(),
                            ),
                        )
                    ],
                    usage=None,
                )
            ]
        )

    def __aiter__(self) -> _FakeOpenAIStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        pass


class _FakeOpenAICompletions:
    async def create(self, **_: Any) -> _FakeOpenAIStream:
        return _FakeOpenAIStream()


async def _assert_model_port_contract(
    case: unittest.TestCase,
    port: Any,
) -> None:
    request = ModelRequest(messages=(UserMessage("hello"),))
    with case.assertRaises(RuntimeError):
        async for _ in port.stream(
            request,
            cancellation=CancellationSource().token,
        ):
            pass

    await port.start()
    events = [
        event
        async for event in port.stream(
            request,
            cancellation=CancellationSource().token,
        )
    ]
    case.assertEqual(events[0], ModelTextDelta("ok"))
    case.assertIsInstance(events[-1], ModelResponseCompleted)
    completed = events[-1]
    assert isinstance(completed, ModelResponseCompleted)
    case.assertEqual(completed.message, AssistantMessage("ok"))
    await port.shutdown()
    with case.assertRaises(RuntimeError):
        async for _ in port.stream(
            request,
            cancellation=CancellationSource().token,
        ):
            pass


class ModelPortContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_and_anthropic_share_kernel_stream_contract(self) -> None:
        openai = OpenAIModelPort(
            ModelConfig("test", "key", "https://openai.invalid"),
            client=SimpleNamespace(
                chat=SimpleNamespace(completions=_FakeOpenAICompletions())
            ),
        )
        anthropic = AnthropicModelPort(
            _anthropic_config(),
            client=FakeAnthropicClient(FakeAnthropicMessages([_text_events()])),
        )

        for port in (openai, anthropic):
            with self.subTest(port=type(port).__name__):
                await _assert_model_port_contract(self, port)


if __name__ == "__main__":
    unittest.main()
