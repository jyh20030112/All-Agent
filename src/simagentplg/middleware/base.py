from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simagentplg.agent.types import StepOutcome

if TYPE_CHECKING:
    from simagentplg.agent.cancellation import CancellationToken
    from simagentplg.agent.state import AgentState
    from simagentplg.agent.types import ToolProgressReporter
    from simagentplg.handlers.definition import ToolDefinition


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    """Metadata and cancellation signal for one tool execution."""

    state: AgentState
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None
    cancellation: CancellationToken | None = None
    progress: ToolProgressReporter | None = None
    tool_definition: ToolDefinition | None = None


ToolNext = Callable[[ToolCallContext], Awaitable[StepOutcome]]


class Middleware:
    """Base class for reusable agent middleware."""

    def __init__(self, *, name: str | None = None, enabled: bool = True) -> None:
        self.name = name or type(self).__name__
        self.enabled = enabled

    async def startup(self) -> None:
        """Initialize optional middleware resources."""

    async def shutdown(self) -> None:
        """Release optional middleware resources."""

    async def on_task_start(self) -> None:
        """Prepare middleware state for one new agent task."""


class ToolMiddleware(Middleware):
    """Decorator around one tool execution."""

    async def __call__(
        self,
        context: ToolCallContext,
        call_next: ToolNext,
    ) -> StepOutcome:
        """Invoke the next decorator or handler in the execution chain."""

        return await call_next(context)


def compose_tool_middlewares(
    middlewares: Sequence[ToolMiddleware],
    terminal: ToolNext,
) -> ToolNext:
    """Wrap a tool terminal with middleware in declaration order."""

    call_next = terminal
    for middleware in reversed(middlewares):
        next_in_chain = call_next

        async def wrapped(
            context: ToolCallContext,
            *,
            middleware: ToolMiddleware = middleware,
            call_next: ToolNext = next_in_chain,
        ) -> StepOutcome:
            return await middleware(context, call_next)

        call_next = wrapped
    return call_next


def format_tool_call_preview(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    risk: str | None = None,
    review: str | None = None,
) -> str:
    """Build a readable approval preview from one tool call."""

    try:
        payload = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
    except TypeError:
        payload = repr(dict(arguments))
    review_text = review or risk
    if review_text:
        return f"Tool: {tool_name}\nReview: {review_text}\nArguments:\n{payload}"
    return f"Tool: {tool_name}\nArguments:\n{payload}"
