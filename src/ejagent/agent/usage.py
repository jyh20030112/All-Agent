from __future__ import annotations

from dataclasses import dataclass

from ejagent.providers.base import ModelUsage


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Aggregated reported usage and coverage for one agent run."""

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


class UsageAccumulator:
    """Mutable per-run collector producing immutable usage snapshots."""

    def __init__(self) -> None:
        self._request_count = 0
        self._reported_request_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens: int | None = None
        self._cache_write_tokens: int | None = None
        self._reasoning_tokens: int | None = None

    def begin_request(self) -> None:
        self._request_count += 1

    def record(self, usage: ModelUsage | None) -> None:
        if usage is None:
            return
        if self._reported_request_count >= self._request_count:
            raise RuntimeError("usage was recorded without an active request")

        previous_reports = self._reported_request_count
        self._reported_request_count += 1
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        self._cache_read_tokens = self._add_optional(
            self._cache_read_tokens,
            usage.cache_read_tokens,
            previous_reports,
        )
        self._cache_write_tokens = self._add_optional(
            self._cache_write_tokens,
            usage.cache_write_tokens,
            previous_reports,
        )
        self._reasoning_tokens = self._add_optional(
            self._reasoning_tokens,
            usage.reasoning_tokens,
            previous_reports,
        )

    def snapshot(self) -> RunUsage:
        return RunUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._input_tokens + self._output_tokens,
            request_count=self._request_count,
            reported_request_count=self._reported_request_count,
            cache_read_tokens=self._cache_read_tokens,
            cache_write_tokens=self._cache_write_tokens,
            reasoning_tokens=self._reasoning_tokens,
        )

    @staticmethod
    def _add_optional(
        current: int | None,
        value: int | None,
        previous_reports: int,
    ) -> int | None:
        if previous_reports == 0:
            return value
        if current is None or value is None:
            return None
        return current + value
