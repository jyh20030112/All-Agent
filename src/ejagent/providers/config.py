from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


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
