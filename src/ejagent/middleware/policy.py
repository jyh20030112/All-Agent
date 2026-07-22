from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from ejagent.agent.cancellation import AgentCancelledError
from ejagent.agent.types import StepOutcome, ToolControl
from ejagent.handlers.definition import ToolEffect
from ejagent.middleware.base import ToolCallContext, ToolMiddleware, ToolNext

logger = logging.getLogger(__name__)

ToolPolicyPredicate = Callable[[ToolCallContext], bool | Awaitable[bool]]


class ToolPolicyAction(StrEnum):
    """One enforceable decision returned by a tool execution policy."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True, slots=True)
class ToolPolicyDecision:
    """Detached policy decision for one tool execution attempt."""

    action: ToolPolicyAction
    reason: str | None = None
    rule_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, ToolPolicyAction):
            raise TypeError("tool policy action must be a ToolPolicyAction")
        _validate_optional_text(self.reason, "tool policy reason")
        _validate_optional_text(self.rule_id, "tool policy rule_id")


@dataclass(frozen=True, slots=True)
class ToolApprovalRequest:
    """Immutable application-facing request for one policy approval."""

    tool_name: str
    arguments: Mapping[str, Any]
    tool_call_id: str | None
    reason: str | None
    rule_id: str | None

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("approval tool_name must not be empty")
        _validate_optional_text(self.tool_call_id, "approval tool_call_id")
        _validate_optional_text(self.reason, "approval reason")
        _validate_optional_text(self.rule_id, "approval rule_id")
        object.__setattr__(self, "arguments", _freeze_value(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolApprovalDecision:
    """Application decision for one requested tool approval."""

    approved: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise TypeError("approval decision must use a boolean approved value")
        _validate_optional_text(self.reason, "approval decision reason")


class ToolApprover(Protocol):
    """Application-owned asynchronous approval boundary."""

    async def approve(self, request: ToolApprovalRequest) -> ToolApprovalDecision:
        """Approve or reject one policy-gated tool execution."""


class ToolExecutionPolicy(ABC):
    """Decision source consumed by :class:`ToolPolicyMiddleware`."""

    async def startup(self) -> None:
        """Initialize optional policy resources."""

    async def shutdown(self) -> None:
        """Release optional policy resources."""

    async def on_task_start(self) -> None:
        """Reset policy state for a newly starting Agent Run."""

    @abstractmethod
    async def evaluate(self, context: ToolCallContext) -> ToolPolicyDecision:
        """Return the decision for one normalized tool execution attempt."""


@dataclass(frozen=True, slots=True)
class ToolPolicyRule:
    """One ordered exact-name/effect rule in a rule-based policy."""

    rule_id: str
    action: ToolPolicyAction
    tool_names: frozenset[str] | None = None
    effects: frozenset[ToolEffect] | None = None
    when: ToolPolicyPredicate | None = None
    max_calls_per_run: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_required_text(self.rule_id, "tool policy rule_id")
        if not isinstance(self.action, ToolPolicyAction):
            raise TypeError("tool policy rule action must be a ToolPolicyAction")
        if self.tool_names is not None:
            names = frozenset(self.tool_names)
            if not names:
                raise ValueError("tool policy rule tool_names must not be empty")
            for name in names:
                _validate_required_text(name, "tool policy rule tool name")
            object.__setattr__(self, "tool_names", names)
        if self.effects is not None:
            effects = frozenset(self.effects)
            if not effects:
                raise ValueError("tool policy rule effects must not be empty")
            if any(not isinstance(effect, ToolEffect) for effect in effects):
                raise TypeError(
                    "tool policy rule effects must contain ToolEffect values"
                )
            object.__setattr__(self, "effects", effects)
        if self.when is not None and not callable(self.when):
            raise TypeError("tool policy rule when must be callable")
        if self.max_calls_per_run is not None:
            if isinstance(self.max_calls_per_run, bool) or not isinstance(
                self.max_calls_per_run, int
            ):
                raise TypeError("max_calls_per_run must be an integer")
            if self.max_calls_per_run <= 0:
                raise ValueError("max_calls_per_run must be greater than zero")
            if self.action is ToolPolicyAction.DENY:
                raise ValueError("a deny rule cannot define max_calls_per_run")
        _validate_optional_text(self.reason, "tool policy rule reason")

    async def matches(self, context: ToolCallContext) -> bool:
        """Return whether all configured selectors match one call."""

        if self.tool_names is not None and context.tool_name not in self.tool_names:
            return False
        effect = (
            context.tool_definition.effect
            if context.tool_definition is not None
            else ToolEffect.SIDE_EFFECTING
        )
        if self.effects is not None and effect not in self.effects:
            return False
        if self.when is None:
            return True
        matched = self.when(context)
        if inspect.isawaitable(matched):
            matched = await matched
        if not isinstance(matched, bool):
            raise TypeError("tool policy rule predicate must return a boolean")
        return matched


class RuleBasedToolPolicy(ToolExecutionPolicy):
    """Evaluate ordered rules with atomic per-Run call reservations."""

    def __init__(
        self,
        rules: Iterable[ToolPolicyRule],
        *,
        default_action: ToolPolicyAction = ToolPolicyAction.DENY,
        default_reason: str | None = None,
    ) -> None:
        self.rules = tuple(rules)
        if any(not isinstance(rule, ToolPolicyRule) for rule in self.rules):
            raise TypeError("rules must contain ToolPolicyRule values")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("tool policy rule_id values must be unique")
        if not isinstance(default_action, ToolPolicyAction):
            raise TypeError("default_action must be a ToolPolicyAction")
        _validate_optional_text(default_reason, "default policy reason")
        self.default_action = default_action
        self.default_reason = default_reason
        self._call_counts: dict[str, int] = {}
        self._count_lock = asyncio.Lock()

    async def on_task_start(self) -> None:
        async with self._count_lock:
            self._call_counts.clear()

    async def evaluate(self, context: ToolCallContext) -> ToolPolicyDecision:
        for rule in self.rules:
            if not await rule.matches(context):
                continue
            if rule.max_calls_per_run is not None:
                async with self._count_lock:
                    count = self._call_counts.get(rule.rule_id, 0)
                    if count >= rule.max_calls_per_run:
                        return ToolPolicyDecision(
                            ToolPolicyAction.DENY,
                            reason=(
                                f"tool call limit reached for policy rule "
                                f"{rule.rule_id!r}"
                            ),
                            rule_id=rule.rule_id,
                        )
                    self._call_counts[rule.rule_id] = count + 1
            return ToolPolicyDecision(
                rule.action,
                reason=rule.reason,
                rule_id=rule.rule_id,
            )
        return ToolPolicyDecision(
            self.default_action,
            reason=self.default_reason,
        )


class ToolPolicyMiddleware(ToolMiddleware):
    """Enforce one tool execution policy through the standard middleware chain."""

    def __init__(
        self,
        policy: ToolExecutionPolicy,
        *,
        approver: ToolApprover | None = None,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        if not isinstance(policy, ToolExecutionPolicy):
            raise TypeError("policy must be a ToolExecutionPolicy")
        super().__init__(name=name, enabled=enabled)
        self.policy = policy
        self.approver = approver

    async def startup(self) -> None:
        await self.policy.startup()

    async def shutdown(self) -> None:
        await self.policy.shutdown()

    async def on_task_start(self) -> None:
        await self.policy.on_task_start()

    async def __call__(
        self,
        context: ToolCallContext,
        call_next: ToolNext,
    ) -> StepOutcome:
        try:
            decision: object = await self.policy.evaluate(context)
        except AgentCancelledError:
            raise
        except Exception:
            logger.exception("Tool policy evaluation failed closed")
            return _rejected_outcome(
                context,
                reason="tool policy evaluation failed",
            )

        if not isinstance(decision, ToolPolicyDecision):
            logger.error(
                "Tool policy returned %s instead of ToolPolicyDecision",
                type(decision).__name__,
            )
            return _rejected_outcome(
                context,
                reason="tool policy returned an invalid decision",
            )

        if decision.action is ToolPolicyAction.ALLOW:
            return await call_next(context)
        if decision.action is ToolPolicyAction.DENY:
            return _rejected_outcome(
                context,
                reason=decision.reason or "tool execution denied by policy",
                rule_id=decision.rule_id,
            )
        return await self._request_approval(context, decision, call_next)

    async def _request_approval(
        self,
        context: ToolCallContext,
        decision: ToolPolicyDecision,
        call_next: ToolNext,
    ) -> StepOutcome:
        if self.approver is None:
            return _rejected_outcome(
                context,
                reason="tool execution requires approval but no approver is configured",
                rule_id=decision.rule_id,
            )
        request = ToolApprovalRequest(
            tool_name=context.tool_name,
            arguments=context.arguments,
            tool_call_id=context.tool_call_id,
            reason=decision.reason,
            rule_id=decision.rule_id,
        )
        try:
            approval: object = await self.approver.approve(request)
        except AgentCancelledError:
            raise
        except Exception:
            logger.exception("Tool approval failed closed")
            return _rejected_outcome(
                context,
                reason="tool approval failed",
                rule_id=decision.rule_id,
            )
        if not isinstance(approval, ToolApprovalDecision):
            logger.error(
                "Tool approver returned %s instead of ToolApprovalDecision",
                type(approval).__name__,
            )
            return _rejected_outcome(
                context,
                reason="tool approver returned an invalid decision",
                rule_id=decision.rule_id,
            )
        if approval.approved:
            return await call_next(context)
        return _rejected_outcome(
            context,
            reason=(
                approval.reason
                or decision.reason
                or "tool execution approval was denied"
            ),
            rule_id=decision.rule_id,
        )


def _rejected_outcome(
    context: ToolCallContext,
    *,
    reason: str,
    rule_id: str | None = None,
) -> StepOutcome:
    data: dict[str, Any] = {
        "status": "rejected",
        "tool": context.tool_name,
        "reason": reason,
    }
    if rule_id is not None:
        data["rule_id"] = rule_id
    return StepOutcome(data, control=ToolControl.REJECT)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _validate_required_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _validate_optional_text(value: object, field: str) -> None:
    if value is not None:
        _validate_required_text(value, field)
