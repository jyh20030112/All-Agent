from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class AnthropicConfig:
    """Connection and generation settings for Anthropic Messages."""

    model: str
    api_key: str
    max_tokens: int = 4096
    base_url: str | None = None
    timeout: float = 60.0
    temperature: float = 0.7

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if self.base_url is not None and not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be numeric")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError("temperature must be numeric")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between zero and one")

    @classmethod
    def from_env(cls) -> AnthropicConfig:
        """Build a config from Anthropic-specific environment variables."""

        load_dotenv()
        model = os.getenv("ANTHROPIC_MODEL")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not model or not api_key:
            raise ValueError("ANTHROPIC_MODEL and ANTHROPIC_API_KEY must be defined")

        try:
            max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
            timeout = float(os.getenv("ANTHROPIC_TIMEOUT", "60"))
            temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))
        except ValueError as exc:
            raise ValueError(
                "ANTHROPIC_MAX_TOKENS, ANTHROPIC_TIMEOUT, and "
                "ANTHROPIC_TEMPERATURE must be numeric"
            ) from exc

        return cls(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            timeout=timeout,
            temperature=temperature,
        )
