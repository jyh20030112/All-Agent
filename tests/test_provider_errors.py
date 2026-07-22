import unittest

import httpx
from openai import APITimeoutError, AuthenticationError, RateLimitError

from ejagent import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelErrorKind,
    ModelProviderError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from ejagent.providers.openai import _normalize_provider_error


class StructuredProviderError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ProviderErrorTests(unittest.TestCase):
    def test_structured_context_code_is_normalized(self) -> None:
        error = _normalize_provider_error(
            StructuredProviderError(
                "context_length_exceeded",
                "request rejected",
            ),
            operation="chat completion",
        )

        self.assertIsInstance(error, ContextOverflowError)
        self.assertEqual(error.kind, ModelErrorKind.CONTEXT_OVERFLOW)

    def test_compatible_context_message_fallback_is_centralized(self) -> None:
        error = _normalize_provider_error(
            RuntimeError("maximum context length is 128000 tokens"),
            operation="chat completion stream",
        )

        self.assertIsInstance(error, ContextOverflowError)

    def test_openai_rate_auth_and_timeout_errors_are_normalized(self) -> None:
        request = httpx.Request("POST", "https://example.invalid/chat")
        auth_response = httpx.Response(401, request=request)
        rate_response = httpx.Response(429, request=request)
        cases = [
            (
                AuthenticationError(
                    "bad key",
                    response=auth_response,
                    body=None,
                ),
                ModelAuthenticationError,
                ModelErrorKind.AUTHENTICATION,
            ),
            (
                RateLimitError(
                    "slow down",
                    response=rate_response,
                    body=None,
                ),
                ModelRateLimitError,
                ModelErrorKind.RATE_LIMIT,
            ),
            (
                APITimeoutError(request),
                ModelTimeoutError,
                ModelErrorKind.TIMEOUT,
            ),
        ]

        for source, error_type, kind in cases:
            with self.subTest(kind=kind):
                error = _normalize_provider_error(
                    source,
                    operation="chat completion",
                )
                self.assertIsInstance(error, error_type)
                self.assertEqual(error.kind, kind)

    def test_unknown_error_remains_provider_error(self) -> None:
        error = _normalize_provider_error(
            ValueError("unexpected response"),
            operation="chat completion",
        )

        self.assertIs(type(error), ModelProviderError)
        self.assertEqual(error.kind, ModelErrorKind.PROVIDER_ERROR)


if __name__ == "__main__":
    unittest.main()
