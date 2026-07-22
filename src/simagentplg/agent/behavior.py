from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from simagentplg.agent.cancellation import CancellationToken
from simagentplg.agent.types import AgentMessage
from simagentplg.agent.usage import RunUsage
from simagentplg.providers.base import AssistantMessage


class BehaviorAction(StrEnum):
    """Action selected by one behavior decision point."""

    CONTINUE = "continue"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class BehaviorDecision:
    """Typed instruction returned by a Behavior Hook."""

    action: BehaviorAction = BehaviorAction.CONTINUE
    output: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, BehaviorAction):
            raise TypeError("action must be a BehaviorAction")
        if self.action is BehaviorAction.CONTINUE and self.output is not None:
            raise ValueError("Continue behavior decision must not contain output")

    @classmethod
    def continue_run(cls) -> BehaviorDecision:
        """Allow the Agent Loop to start another Turn."""

        return cls()

    @classmethod
    def stop(cls, output: str | None = None) -> BehaviorDecision:
        """Complete the Run at the current full-Turn boundary."""

        return cls(action=BehaviorAction.STOP, output=output)


@dataclass(frozen=True, slots=True, init=False)
class TurnSnapshot:
    """Detached read snapshot exposed at the after-Turn safe point."""

    agent_id: str
    run_id: str
    turn: int
    task: str | None
    response: AssistantMessage
    usage: RunUsage
    _messages: tuple[AgentMessage, ...]

    def __init__(
        self,
        *,
        agent_id: str,
        run_id: str,
        turn: int,
        task: str | None,
        response: AssistantMessage,
        usage: RunUsage,
        messages: Sequence[Mapping[str, object]],
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        if not run_id:
            raise ValueError("run_id must not be empty")
        if turn <= 0:
            raise ValueError("turn must be greater than zero")
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "turn", turn)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(
            self,
            "_messages",
            tuple(deepcopy(dict(message)) for message in messages),
        )

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """Return a fresh detached transcript for this completed Turn."""

        return tuple(deepcopy(message) for message in self._messages)


class BehaviorHook(Protocol):
    """Behavior extension evaluated after a non-terminal full Turn."""

    async def after_turn(
        self,
        snapshot: TurnSnapshot,
        *,
        cancellation: CancellationToken,
    ) -> BehaviorDecision | None:
        """Return STOP or allow the next Turn with CONTINUE/None."""


class BehaviorHookError(RuntimeError):
    """One Behavior Hook failed or returned an unsupported decision."""

    def __init__(self, hook: BehaviorHook, error: BaseException) -> None:
        self.hook = hook
        self.error = error
        hook_name = getattr(hook, "name", type(hook).__name__)
        super().__init__(f"behavior hook {hook_name!r} failed: {error}")
