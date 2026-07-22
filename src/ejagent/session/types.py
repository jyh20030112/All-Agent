from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum

from ejagent.agent.compaction import SummaryEntry
from ejagent.agent.result import AgentRunResult
from ejagent.agent.types import AgentMessage


class SessionRunIntent(StrEnum):
    """Why one Session Run entered the agent runtime."""

    TASK = "task"
    CONTINUE = "continue"


@dataclass(frozen=True, slots=True)
class SessionMessage:
    """One persistent conversation message associated with an agent run."""

    run_id: str
    sequence: int
    message: AgentMessage

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.sequence <= 0:
            raise ValueError("sequence must be greater than zero")
        object.__setattr__(self, "message", deepcopy(self.message))


@dataclass(frozen=True, slots=True)
class SessionRun:
    """Persistent boundary and terminal result for one agent run."""

    run_id: str
    task: str | None
    start_sequence: int
    intent: SessionRunIntent = SessionRunIntent.TASK
    finish_sequence: int | None = None
    result: AgentRunResult | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.start_sequence <= 0:
            raise ValueError("start_sequence must be greater than zero")
        if not isinstance(self.intent, SessionRunIntent):
            raise TypeError("intent must be a SessionRunIntent")
        if self.intent is SessionRunIntent.TASK:
            if self.task is None:
                raise ValueError("task Run requires a task")
        elif self.task is not None:
            raise ValueError("Continue Run must not contain a task")
        if (self.finish_sequence is None) != (self.result is None):
            raise ValueError("finish_sequence and result must be set together")
        if (
            self.finish_sequence is not None
            and self.finish_sequence <= self.start_sequence
        ):
            raise ValueError("finish_sequence must be greater than start_sequence")

    @property
    def finished(self) -> bool:
        return self.result is not None


@dataclass(frozen=True, slots=True)
class SessionCompaction:
    """One compacted conversation projection with retained audit history."""

    operation_id: str
    sequence: int
    summary: SummaryEntry
    messages: tuple[AgentMessage, ...]
    covered_entry_count: int

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must not be empty")
        if self.sequence <= 0:
            raise ValueError("sequence must be greater than zero")
        if not self.messages:
            raise ValueError("compaction messages must not be empty")
        if self.covered_entry_count < 0:
            raise ValueError("covered_entry_count must not be negative")
        object.__setattr__(self, "messages", deepcopy(self.messages))


@dataclass(slots=True)
class AgentSession:
    """Linear conversation history spanning one or more agent runs."""

    session_id: str
    agent_id: str | None = None
    entries: list[SessionMessage] = field(default_factory=list)
    runs: list[SessionRun] = field(default_factory=list)
    compactions: list[SessionCompaction] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session_id = self.session_id.strip()
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.agent_id is not None:
            self.agent_id = self.agent_id.strip()
            if not self.agent_id:
                raise ValueError("agent_id must not be empty")
        self.entries = [
            SessionMessage(
                run_id=entry.run_id,
                sequence=entry.sequence,
                message=entry.message,
            )
            for entry in self.entries
        ]
        self.runs = list(self.runs)
        self.compactions = [
            SessionCompaction(
                operation_id=compaction.operation_id,
                sequence=compaction.sequence,
                summary=compaction.summary,
                messages=compaction.messages,
                covered_entry_count=compaction.covered_entry_count,
            )
            for compaction in self.compactions
        ]
        if any(
            compaction.covered_entry_count > len(self.entries)
            for compaction in self.compactions
        ):
            raise ValueError("compaction covers unavailable session entries")

    @property
    def messages(self) -> list[AgentMessage]:
        """Return an independent history suitable for ``BaseAgent.reset``."""

        if not self.compactions:
            return [deepcopy(entry.message) for entry in self.entries]

        latest = self.compactions[-1]
        compacted = [deepcopy(message) for message in latest.messages]
        compacted.extend(
            deepcopy(entry.message)
            for entry in self.entries[latest.covered_entry_count :]
        )
        return compacted

    def bind_agent(self, agent_id: str) -> None:
        """Bind a new Session to one logical agent identity."""

        agent_id = agent_id.strip()
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        if self.agent_id is None:
            self.agent_id = agent_id
        elif self.agent_id != agent_id:
            raise ValueError(
                f"session {self.session_id!r} belongs to agent "
                f"{self.agent_id!r}, not {agent_id!r}"
            )

    def begin_run(self, run_id: str, task: str, sequence: int) -> None:
        """Open one run and persist its user task message."""

        if any(run.run_id == run_id for run in self.runs):
            raise ValueError(f"run {run_id!r} already exists")
        if any(not run.finished for run in self.runs):
            raise ValueError("session already has an unfinished run")
        run = SessionRun(
            run_id=run_id,
            task=task,
            start_sequence=sequence,
            intent=SessionRunIntent.TASK,
        )
        self.runs.append(run)
        self.append_message(
            run_id,
            sequence,
            {"role": "user", "content": task},
        )

    def begin_continue(self, run_id: str, sequence: int) -> None:
        """Open one Continue Run without appending a user message."""

        if any(run.run_id == run_id for run in self.runs):
            raise ValueError(f"run {run_id!r} already exists")
        if any(not run.finished for run in self.runs):
            raise ValueError("session already has an unfinished run")
        self.runs.append(
            SessionRun(
                run_id=run_id,
                task=None,
                start_sequence=sequence,
                intent=SessionRunIntent.CONTINUE,
            )
        )

    def append_message(
        self,
        run_id: str,
        sequence: int,
        message: AgentMessage,
    ) -> None:
        """Append a persistent message to an unfinished run."""

        run = self._get_run(run_id)
        if run.finished:
            raise ValueError(f"run {run_id!r} is already finished")
        if sequence < self._last_sequence(run_id):
            raise ValueError(f"event sequence moved backwards for run {run_id!r}")
        self.entries.append(
            SessionMessage(
                run_id=run_id,
                sequence=sequence,
                message=message,
            )
        )

    def finish_run(
        self,
        run_id: str,
        sequence: int,
        result: AgentRunResult,
    ) -> None:
        """Attach the existing structured result to an unfinished run."""

        index = self._run_index(run_id)
        run = self.runs[index]
        if run.finished:
            raise ValueError(f"run {run_id!r} is already finished")
        if sequence <= self._last_sequence(run_id):
            raise ValueError(f"finish sequence must follow messages for run {run_id!r}")
        self.runs[index] = replace(
            run,
            finish_sequence=sequence,
            result=result,
        )

    def apply_compaction(
        self,
        operation_id: str,
        sequence: int,
        summary: SummaryEntry,
        messages: tuple[AgentMessage, ...],
    ) -> None:
        """Record a new compacted projection without deleting audit entries."""

        if any(
            compaction.operation_id == operation_id for compaction in self.compactions
        ):
            raise ValueError(f"compaction operation {operation_id!r} already exists")
        self.compactions.append(
            SessionCompaction(
                operation_id=operation_id,
                sequence=sequence,
                summary=summary,
                messages=messages,
                covered_entry_count=len(self.entries),
            )
        )

    def snapshot(self) -> AgentSession:
        """Return a detached copy safe for storage and callers."""

        return AgentSession(
            session_id=self.session_id,
            agent_id=self.agent_id,
            entries=list(self.entries),
            runs=list(self.runs),
            compactions=list(self.compactions),
        )

    def _get_run(self, run_id: str) -> SessionRun:
        return self.runs[self._run_index(run_id)]

    def _run_index(self, run_id: str) -> int:
        for index, run in enumerate(self.runs):
            if run.run_id == run_id:
                return index
        raise ValueError(f"unknown run {run_id!r}")

    def _last_sequence(self, run_id: str) -> int:
        run = self._get_run(run_id)
        return max(
            (entry.sequence for entry in self.entries if entry.run_id == run_id),
            default=run.start_sequence,
        )
