from __future__ import annotations

from ejagent.contracts.usage import RunUsage
from ejagent.providers.base import ModelUsage


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
