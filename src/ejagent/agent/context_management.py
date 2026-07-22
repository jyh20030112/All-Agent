from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from ejagent.agent.types import AgentMessage, is_internal_metadata_key
from ejagent.providers.base import ModelUsage


class ContextUsageSource(StrEnum):
    """Origin of one context pressure estimate."""

    ESTIMATED = "estimated"
    REPORTED = "reported"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    """Conservative token estimate for one complete provider request."""

    reported_tokens: int
    trailing_tokens: int
    heuristic_tokens: int
    total_tokens: int
    last_usage_index: int | None
    source: ContextUsageSource

    def __post_init__(self) -> None:
        for name in (
            "reported_tokens",
            "trailing_tokens",
            "heuristic_tokens",
            "total_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.last_usage_index is not None and self.last_usage_index < 0:
            raise ValueError("last_usage_index must not be negative")
        if self.total_tokens < self.heuristic_tokens:
            raise ValueError("total_tokens must cover heuristic_tokens")
        if self.total_tokens < self.reported_tokens + self.trailing_tokens:
            raise ValueError("total_tokens must cover reported and trailing tokens")
        if (
            self.last_usage_index is None
            and self.source is not ContextUsageSource.ESTIMATED
        ):
            raise ValueError("an estimate without usage must use estimated source")

    @property
    def usage_based_tokens(self) -> int:
        """Return the reported baseline plus messages added after it."""

        return self.reported_tokens + self.trailing_tokens


class MessageTokenEstimator(Protocol):
    """Replaceable token estimator for provider-visible context values."""

    def estimate_message(self, message: Mapping[str, Any]) -> int:
        """Estimate one provider-visible message."""

    def estimate_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        """Estimate the complete tool definition collection."""


class HeuristicMessageTokenEstimator:
    """UTF-8-aware fallback used when no provider tokenizer is available."""

    message_overhead_tokens = 4
    tool_overhead_tokens = 8

    def estimate_message(self, message: Mapping[str, Any]) -> int:
        visible = {
            key: value
            for key, value in message.items()
            if key != "usage" and not is_internal_metadata_key(key)
        }
        return self.message_overhead_tokens + self._estimate_value(visible)

    def estimate_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        if not tools:
            return 0
        return self.tool_overhead_tokens + self._estimate_value(list(tools))

    @staticmethod
    def _estimate_value(value: Any) -> int:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        ascii_count = sum(ord(character) < 128 for character in serialized)
        non_ascii_count = len(serialized) - ascii_count
        return math.ceil(ascii_count / 4 + non_ascii_count)


def estimate_context_usage(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] = (),
    estimator: MessageTokenEstimator | None = None,
) -> ContextUsageEstimate:
    """Estimate the full request from the latest Usage plus a safe fallback.

    The most recent assistant Usage is the best baseline for the repeated
    context. Messages appended after that response are estimated. A complete
    heuristic pass, including tool definitions, guards against changed
    projections and is used as a lower bound.
    """

    active_estimator = estimator or HeuristicMessageTokenEstimator()
    heuristic_tokens = sum(
        active_estimator.estimate_message(message) for message in messages
    ) + active_estimator.estimate_tools(tools)

    usage_info = _last_usage_info(messages)
    if usage_info is None:
        return ContextUsageEstimate(
            reported_tokens=0,
            trailing_tokens=heuristic_tokens,
            heuristic_tokens=heuristic_tokens,
            total_tokens=heuristic_tokens,
            last_usage_index=None,
            source=ContextUsageSource.ESTIMATED,
        )

    usage_index, reported_tokens = usage_info
    trailing_tokens = sum(
        active_estimator.estimate_message(message)
        for message in messages[usage_index + 1 :]
    )
    usage_based_tokens = reported_tokens + trailing_tokens
    total_tokens = max(usage_based_tokens, heuristic_tokens)
    source = (
        ContextUsageSource.REPORTED
        if trailing_tokens == 0 and total_tokens == reported_tokens
        else ContextUsageSource.MIXED
    )
    return ContextUsageEstimate(
        reported_tokens=reported_tokens,
        trailing_tokens=trailing_tokens,
        heuristic_tokens=heuristic_tokens,
        total_tokens=total_tokens,
        last_usage_index=usage_index,
        source=source,
    )


def _last_usage_info(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[int, int] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant" or "usage" not in message:
            continue
        total_tokens = _usage_total_tokens(message["usage"])
        if total_tokens is not None:
            return index, total_tokens
    return None


def _usage_total_tokens(value: Any) -> int | None:
    if isinstance(value, ModelUsage):
        return value.total_tokens
    if not isinstance(value, Mapping):
        return None
    total_tokens = value.get("total_tokens")
    if (
        isinstance(total_tokens, int)
        and not isinstance(total_tokens, bool)
        and total_tokens >= 0
    ):
        return total_tokens
    return None


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Capacity reserved for one model request, independent of run spend."""

    context_window: int
    reserve_tokens: int
    keep_recent_tokens: int

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError("context_window must be greater than zero")
        if self.reserve_tokens < 0:
            raise ValueError("reserve_tokens must not be negative")
        if self.reserve_tokens >= self.context_window:
            raise ValueError("reserve_tokens must be less than context_window")
        if self.keep_recent_tokens <= 0:
            raise ValueError("keep_recent_tokens must be greater than zero")
        if self.keep_recent_tokens > self.threshold_tokens:
            raise ValueError("keep_recent_tokens must not exceed the context threshold")

    @property
    def threshold_tokens(self) -> int:
        """Return the pressure threshold before reserved capacity."""

        return self.context_window - self.reserve_tokens


class CompactionDecisionReason(StrEnum):
    """Reason behind one compaction policy decision."""

    DISABLED = "disabled"
    BELOW_THRESHOLD = "below_threshold"
    THRESHOLD_REACHED = "threshold_reached"


@dataclass(frozen=True, slots=True)
class CompactionDecision:
    """Pure policy result derived from a context usage estimate."""

    estimate: ContextUsageEstimate
    threshold_tokens: int
    should_compact: bool
    reason: CompactionDecisionReason

    @property
    def pressure_ratio(self) -> float:
        """Return context pressure relative to the policy threshold."""

        return self.estimate.total_tokens / self.threshold_tokens


@dataclass(frozen=True, slots=True)
class CompactionPreparation:
    """Non-mutating plan for summarizing old persistent conversation turns."""

    protected_messages: tuple[AgentMessage, ...]
    messages_to_summarize: tuple[AgentMessage, ...]
    messages_to_keep: tuple[AgentMessage, ...]
    history_start_index: int
    first_kept_index: int
    estimated_history_tokens: int
    estimated_summarized_tokens: int
    estimated_kept_tokens: int

    @property
    def can_compact(self) -> bool:
        """Return whether at least one complete old turn can be summarized."""

        return bool(self.messages_to_summarize)


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    """Decide when context pressure warrants preparing compaction."""

    budget: ContextBudget
    enabled: bool = True

    def evaluate(
        self,
        estimate: ContextUsageEstimate,
    ) -> CompactionDecision:
        if not self.enabled:
            return CompactionDecision(
                estimate=estimate,
                threshold_tokens=self.budget.threshold_tokens,
                should_compact=False,
                reason=CompactionDecisionReason.DISABLED,
            )
        should_compact = estimate.total_tokens >= self.budget.threshold_tokens
        return CompactionDecision(
            estimate=estimate,
            threshold_tokens=self.budget.threshold_tokens,
            should_compact=should_compact,
            reason=(
                CompactionDecisionReason.THRESHOLD_REACHED
                if should_compact
                else CompactionDecisionReason.BELOW_THRESHOLD
            ),
        )

    def prepare(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        estimator: MessageTokenEstimator | None = None,
    ) -> CompactionPreparation:
        return prepare_compaction(
            messages,
            keep_recent_tokens=self.budget.keep_recent_tokens,
            estimator=estimator,
        )


@dataclass(frozen=True, slots=True)
class AutoCompactionPolicy:
    """Opt-in behavior for proactive compaction and overflow recovery."""

    enabled: bool = True
    compact_on_pressure: bool = True
    recover_on_overflow: bool = True
    max_overflow_retries: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if not isinstance(self.compact_on_pressure, bool):
            raise TypeError("compact_on_pressure must be a bool")
        if not isinstance(self.recover_on_overflow, bool):
            raise TypeError("recover_on_overflow must be a bool")
        if (
            not isinstance(self.max_overflow_retries, int)
            or isinstance(self.max_overflow_retries, bool)
            or self.max_overflow_retries not in {0, 1}
        ):
            raise ValueError("max_overflow_retries must be zero or one")


def prepare_compaction(
    messages: Sequence[Mapping[str, Any]],
    *,
    keep_recent_tokens: int,
    estimator: MessageTokenEstimator | None = None,
) -> CompactionPreparation:
    """Prepare a safe full-turn cut without changing Agent State.

    Leading system messages are protected. Only a contiguous suffix containing
    user, assistant, and tool messages is eligible. Cuts occur immediately
    before a user message, so assistant tool calls and their tool results are
    never separated by this first compaction implementation.
    """

    if keep_recent_tokens <= 0:
        raise ValueError("keep_recent_tokens must be greater than zero")
    active_estimator = estimator or HeuristicMessageTokenEstimator()
    copied = [deepcopy(dict(message)) for message in messages]
    history_start = _compactable_history_start(copied)
    ranges = _turn_ranges(copied, history_start)

    if not ranges:
        return CompactionPreparation(
            protected_messages=tuple(copied),
            messages_to_summarize=(),
            messages_to_keep=(),
            history_start_index=history_start,
            first_kept_index=history_start,
            estimated_history_tokens=0,
            estimated_summarized_tokens=0,
            estimated_kept_tokens=0,
        )

    turn_tokens = [
        sum(active_estimator.estimate_message(message) for message in copied[start:end])
        for start, end in ranges
    ]
    kept_turn_index = len(ranges) - 1
    kept_tokens = turn_tokens[kept_turn_index]
    while kept_turn_index > 0 and kept_tokens < keep_recent_tokens:
        kept_turn_index -= 1
        kept_tokens += turn_tokens[kept_turn_index]

    first_kept = ranges[kept_turn_index][0]
    summarized_tokens = sum(turn_tokens[:kept_turn_index])
    history_tokens = sum(turn_tokens)
    return CompactionPreparation(
        protected_messages=tuple(copied[:history_start]),
        messages_to_summarize=tuple(copied[history_start:first_kept]),
        messages_to_keep=tuple(copied[first_kept:]),
        history_start_index=history_start,
        first_kept_index=first_kept,
        estimated_history_tokens=history_tokens,
        estimated_summarized_tokens=summarized_tokens,
        estimated_kept_tokens=history_tokens - summarized_tokens,
    )


def _compactable_history_start(messages: Sequence[Mapping[str, Any]]) -> int:
    history_start = 0
    while (
        history_start < len(messages)
        and messages[history_start].get("role") == "system"
    ):
        history_start += 1

    compactable_roles = {"user", "assistant", "tool"}
    for index in range(history_start, len(messages)):
        if messages[index].get("role") not in compactable_roles:
            history_start = index + 1
    return history_start


def _turn_ranges(
    messages: Sequence[Mapping[str, Any]],
    history_start: int,
) -> list[tuple[int, int]]:
    if history_start >= len(messages):
        return []

    starts = [history_start]
    for index in range(history_start + 1, len(messages)):
        if messages[index].get("role") == "user":
            starts.append(index)
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(messages))
        for index, start in enumerate(starts)
    ]
