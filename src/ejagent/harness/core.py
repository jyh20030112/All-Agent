from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import TracebackType
from typing import Self
from uuid import uuid4

from ejagent.contracts.context import ContextPipeline
from ejagent.contracts.control import CancellationSource
from ejagent.contracts.conversation import ConversationSnapshot
from ejagent.contracts.json import JsonValue
from ejagent.contracts.lifecycle import ManagedResource
from ejagent.contracts.messages import ConversationMessage
from ejagent.contracts.model import ModelPort
from ejagent.contracts.runs import (
    AuditRecord,
    FailureCode,
    RunFailure,
    RunIntent,
    RunLimits,
    RunOutcome,
    RunPhase,
    RunResult,
    RunSpec,
    RunStatus,
    StopReason,
)
from ejagent.contracts.session import (
    SessionCommit,
    SessionSnapshot,
    SessionStore,
    SessionStoreError,
)
from ejagent.contracts.tools import ToolExecutor
from ejagent.harness._memory import MemorySessionStore
from ejagent.kernel import RuntimeKernel

RunIdFactory = Callable[[], str]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class HarnessStatus(StrEnum):
    """Observable lifecycle phase of one AgentHarness."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    CLOSED = "closed"


class HarnessClosedError(RuntimeError):
    """A Run was requested after Harness shutdown began."""


class SessionStoreProtocolError(RuntimeError):
    """A SessionStore returned a snapshot that violates its contract."""


class AgentHarness:
    """Own one agent's resources, Conversation, and atomic Run commits."""

    def __init__(
        self,
        *,
        agent_id: str,
        model: ModelPort,
        tools: ToolExecutor,
        context: ContextPipeline | None = None,
        initial_messages: Iterable[ConversationMessage] = (),
        store: SessionStore | None = None,
        resources: Iterable[object] = (),
        limits: RunLimits | None = None,
        configuration_revision: str = "default",
        run_id_factory: RunIdFactory | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(agent_id, str):
            raise TypeError("agent_id must be a string")
        agent_id = agent_id.strip()
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        if not isinstance(configuration_revision, str):
            raise TypeError("configuration_revision must be a string")
        if not configuration_revision.strip():
            raise ValueError("configuration_revision must not be empty")
        if limits is not None and not isinstance(limits, RunLimits):
            raise TypeError("limits must be RunLimits or None")
        if run_id_factory is not None and not callable(run_id_factory):
            raise TypeError("run_id_factory must be callable or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")

        initial = SessionSnapshot(
            agent_id=agent_id,
            conversation=ConversationSnapshot(messages=tuple(initial_messages)),
        )
        self._agent_id = agent_id
        self._model = model
        self._tools = tools
        self._context = context
        self._store = store if store is not None else MemorySessionStore()
        self._snapshot = initial
        self._limits = limits or RunLimits()
        self._configuration_revision = configuration_revision
        self._run_id_factory = run_id_factory or (lambda: str(uuid4()))
        self._clock = clock or _utc_now
        self._kernel = RuntimeKernel(
            model=model,
            tools=tools,
            context=context,
            clock=self._clock,
        )
        self._resources = self._managed_resources(
            (self._store, self._model, self._tools, self._context, *resources)
        )
        self._started_resources: tuple[ManagedResource, ...] = ()
        self._status = HarnessStatus.NEW
        self._closing = False
        self._active_cancellation: CancellationSource | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def status(self) -> HarnessStatus:
        return self._status

    @property
    def revision(self) -> int:
        return self._snapshot.revision

    @property
    def messages(self) -> tuple[ConversationMessage, ...]:
        return self._snapshot.messages

    @property
    def last_result(self) -> RunResult | None:
        return self._snapshot.last_result

    @property
    def snapshot(self) -> SessionSnapshot:
        return self._snapshot

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.shutdown()

    async def start(self) -> None:
        """Start owned resources transactionally and restore Conversation state."""

        async with self._lifecycle_lock:
            if self._status in (HarnessStatus.READY, HarnessStatus.RUNNING):
                return
            if self._closing or self._status is HarnessStatus.CLOSED:
                raise HarnessClosedError(f"agent harness {self._agent_id!r} is closed")

            self._status = HarnessStatus.STARTING
            started: list[ManagedResource] = []
            try:
                for resource in self._resources:
                    await resource.start()
                    started.append(resource)
                loaded = await self._store.load(self._agent_id)
                if loaded is not None:
                    self._validate_loaded_snapshot(loaded)
                    self._snapshot = loaded
            except BaseException:
                await self._rollback_start(started)
                self._status = HarnessStatus.NEW
                raise

            self._started_resources = tuple(started)
            self._status = HarnessStatus.READY

    async def run(
        self,
        task: str,
        *,
        limits: RunLimits | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> RunOutcome:
        """Run one task after earlier calls, then atomically commit on success."""

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must not be empty")
        return await self._execute(
            intent=RunIntent.TASK,
            task=task,
            limits=limits,
            metadata=metadata,
        )

    async def continue_run(
        self,
        *,
        limits: RunLimits | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> RunOutcome:
        """Continue committed Conversation without appending a user message."""

        return await self._execute(
            intent=RunIntent.CONTINUE,
            task=None,
            limits=limits,
            metadata=metadata,
        )

    def cancel(self, reason: str | None = None) -> bool:
        """Request cooperative cancellation of the active Run, if any."""

        source = self._active_cancellation
        return source.cancel(reason) if source is not None else False

    async def shutdown(self) -> None:
        """Stop accepting Runs, cancel active work, and release resources."""

        async with self._lifecycle_lock:
            if self._status is HarnessStatus.CLOSED:
                return
            self._closing = True
            self._status = HarnessStatus.STOPPING
            self.cancel("AgentHarness is shutting down")

            async with self._run_lock:
                failures: list[BaseException] = []
                for resource in reversed(self._started_resources):
                    try:
                        await resource.shutdown()
                    except BaseException as exc:
                        failures.append(exc)
                self._started_resources = ()
                self._status = HarnessStatus.CLOSED

            if failures:
                raise BaseExceptionGroup(
                    "one or more AgentHarness resources failed to shut down",
                    failures,
                )

    async def _execute(
        self,
        *,
        intent: RunIntent,
        task: str | None,
        limits: RunLimits | None,
        metadata: Mapping[str, JsonValue] | None,
    ) -> RunOutcome:
        await self.start()
        async with self._run_lock:
            if self._closing or self._status is HarnessStatus.CLOSED:
                raise HarnessClosedError(f"agent harness {self._agent_id!r} is closed")

            base = self._snapshot
            spec = RunSpec(
                run_id=self._run_id_factory(),
                base_revision=base.revision,
                intent=intent,
                task=task,
                messages=base.messages,
                limits=limits or self._limits,
                configuration_revision=self._configuration_revision,
                metadata=metadata or {},
            )
            cancellation = CancellationSource()
            self._active_cancellation = cancellation
            self._status = HarnessStatus.RUNNING
            try:
                outcome = await self._kernel.run(
                    spec,
                    cancellation=cancellation.token,
                )
                return await self._commit(base, outcome)
            finally:
                self._active_cancellation = None
                if not self._closing:
                    self._status = HarnessStatus.READY

    async def _commit(
        self,
        base: SessionSnapshot,
        outcome: RunOutcome,
    ) -> RunOutcome:
        commit = SessionCommit(
            agent_id=self._agent_id,
            base=base.conversation,
            outcome=outcome,
        )
        try:
            snapshot = await self._store.commit(commit)
        except SessionStoreError as exc:
            return self._persistence_failure(outcome, exc)

        self._validate_committed_snapshot(snapshot, commit, base)
        self._snapshot = snapshot
        return outcome

    def _persistence_failure(
        self,
        outcome: RunOutcome,
        error: SessionStoreError,
    ) -> RunOutcome:
        failure = RunFailure(
            phase=RunPhase.COMMIT,
            code=FailureCode.PERSISTENCE_FAILED,
            message=str(error) or type(error).__name__,
            retryable=True,
            cause=error,
        )
        result = RunResult(
            run_id=outcome.result.run_id,
            status=RunStatus.FAILED,
            stop_reason=StopReason.PERSISTENCE_FAILED,
            turns=outcome.result.turns,
            usage=outcome.result.usage,
        )
        audit = (
            *outcome.audit_records,
            AuditRecord(
                run_id=outcome.result.run_id,
                sequence=len(outcome.audit_records) + 1,
                kind="commit_failed",
                occurred_at=self._clock(),
                payload={"error": str(error) or type(error).__name__},
            ),
        )
        return RunOutcome(
            result=result,
            delta=outcome.delta,
            audit_records=audit,
            failure=failure,
        )

    def _validate_loaded_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, SessionSnapshot):
            raise SessionStoreProtocolError(
                "SessionStore.load() must return SessionSnapshot or None"
            )
        if snapshot.agent_id != self._agent_id:
            raise SessionStoreProtocolError(
                f"SessionStore loaded agent {snapshot.agent_id!r} for "
                f"{self._agent_id!r}"
            )

    def _validate_committed_snapshot(
        self,
        snapshot: object,
        commit: SessionCommit,
        base: SessionSnapshot,
    ) -> None:
        if not isinstance(snapshot, SessionSnapshot):
            raise SessionStoreProtocolError(
                "SessionStore.commit() must return SessionSnapshot"
            )
        expected_result = (
            commit.outcome.result if commit.advances_revision else base.last_result
        )
        if (
            snapshot.agent_id != self._agent_id
            or snapshot.revision != commit.resulting_revision
            or snapshot.conversation != commit.resulting_conversation
            or snapshot.last_result != expected_result
        ):
            raise SessionStoreProtocolError(
                "SessionStore.commit() returned a snapshot inconsistent "
                "with the proposed commit"
            )

    async def _rollback_start(self, started: list[ManagedResource]) -> None:
        for resource in reversed(started):
            try:
                await resource.shutdown()
            except BaseException:
                pass

    @staticmethod
    def _managed_resources(
        resources: Iterable[object],
    ) -> tuple[ManagedResource, ...]:
        managed: list[ManagedResource] = []
        seen: set[int] = set()
        for resource in resources:
            identity = id(resource)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(resource, ManagedResource):
                managed.append(resource)
        return tuple(managed)
