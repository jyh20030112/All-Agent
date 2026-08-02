from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Aggregated reported usage and coverage for one Run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    reported_request_count: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "request_count",
            "reported_request_count",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        if self.reported_request_count > self.request_count:
            raise ValueError("reported_request_count must not exceed request_count")
        if (
            self.cache_read_tokens is not None
            and self.cache_read_tokens > self.input_tokens
        ):
            raise ValueError("cache_read_tokens must not exceed input_tokens")
        if (
            self.cache_write_tokens is not None
            and self.cache_write_tokens > self.input_tokens
        ):
            raise ValueError("cache_write_tokens must not exceed input_tokens")
        if (
            self.reasoning_tokens is not None
            and self.reasoning_tokens > self.output_tokens
        ):
            raise ValueError("reasoning_tokens must not exceed output_tokens")

    @property
    def complete(self) -> bool:
        """Return whether every attempted model request reported usage."""

        return self.reported_request_count == self.request_count

    @property
    def missing_request_count(self) -> int:
        return self.request_count - self.reported_request_count

    def to_dict(self) -> dict[str, int | None]:
        """Return a detached JSON-compatible representation."""

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "reported_request_count": self.reported_request_count,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
