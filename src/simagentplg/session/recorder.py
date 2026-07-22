import asyncio

from simagentplg.agent.events import (
    AgentContinued,
    AgentEvent,
    AgentFinished,
    AgentStarted,
    CompactionCompleted,
    MessageCompleted,
    SteeringApplied,
    ToolCompleted,
)
from simagentplg.providers.base import serialize_assistant_message
from simagentplg.session.journal import DEFAULT_SESSION_BRANCH, SessionRecordDraft
from simagentplg.session.storage import SessionJournalStorage, SessionStorage
from simagentplg.session.types import AgentSession

_RECORDED_PAYLOADS = (
    AgentStarted,
    AgentContinued,
    MessageCompleted,
    SteeringApplied,
    ToolCompleted,
    AgentFinished,
    CompactionCompleted,
)


class SessionRecorder:
    """Build one selected Session branch from read-only lifecycle events."""

    def __init__(
        self,
        *,
        session_id: str,
        storage: SessionStorage,
        branch_id: str = DEFAULT_SESSION_BRANCH,
    ) -> None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        self.session_id = session_id
        self.storage = storage
        self.branch_id = branch_id.strip()
        if not self.branch_id:
            raise ValueError("branch_id must not be empty")
        self._lock = asyncio.Lock()

    async def emit(self, event: AgentEvent) -> None:
        payload = event.payload
        if not isinstance(payload, _RECORDED_PAYLOADS):
            return
        if isinstance(payload, CompactionCompleted) and not payload.result.completed:
            return

        async with self._lock:
            expected_head_id: str | None = None
            journal_storage = (
                self.storage
                if isinstance(self.storage, SessionJournalStorage)
                else None
            )
            if journal_storage is not None:
                checkout = await journal_storage.checkout(
                    self.session_id,
                    branch_id=self.branch_id,
                )
                session = checkout.session if checkout is not None else None
                expected_head_id = (
                    checkout.head.record_id if checkout is not None else None
                )
            else:
                session = await self.storage.load(self.session_id)
            if session is None:
                if self.branch_id != DEFAULT_SESSION_BRANCH:
                    raise ValueError(f"unknown Session branch {self.branch_id!r}")
                session = AgentSession(session_id=self.session_id)
            session.bind_agent(event.agent_id)

            if isinstance(payload, AgentStarted):
                session.begin_run(event.run_id, payload.task, event.sequence)
                draft = SessionRecordDraft.run_started(
                    session_id=self.session_id,
                    agent_id=event.agent_id,
                    sequence=event.sequence,
                    run_id=event.run_id,
                    task=payload.task,
                    branch_id=self.branch_id,
                )
            elif isinstance(payload, AgentContinued):
                session.begin_continue(event.run_id, event.sequence)
                draft = SessionRecordDraft.run_continued(
                    session_id=self.session_id,
                    agent_id=event.agent_id,
                    sequence=event.sequence,
                    run_id=event.run_id,
                    branch_id=self.branch_id,
                )
            elif isinstance(payload, SteeringApplied):
                session.append_message(
                    event.run_id,
                    event.sequence,
                    {"role": "user", "content": payload.control.content},
                )
                draft = SessionRecordDraft.steering_applied(
                    session_id=self.session_id,
                    agent_id=event.agent_id,
                    sequence=event.sequence,
                    run_id=event.run_id,
                    input_id=payload.control.input_id,
                    content=payload.control.content,
                    target_turn=payload.target_turn,
                    branch_id=self.branch_id,
                )
            elif isinstance(payload, CompactionCompleted):
                assert payload.result.summary is not None
                session.apply_compaction(
                    payload.result.operation_id,
                    event.sequence,
                    payload.result.summary,
                    payload.result.messages,
                )
                draft = SessionRecordDraft.compaction_applied(
                    session_id=self.session_id,
                    agent_id=event.agent_id,
                    sequence=event.sequence,
                    result=payload.result,
                    branch_id=self.branch_id,
                )
            elif isinstance(payload, MessageCompleted):
                message = serialize_assistant_message(
                    payload.message,
                    usage=payload.usage,
                )
                session.append_message(
                    event.run_id,
                    event.sequence,
                    message,
                )
                draft = SessionRecordDraft.message_appended(
                    session_id=self.session_id,
                    agent_id=event.agent_id,
                    sequence=event.sequence,
                    run_id=event.run_id,
                    message=message,
                    branch_id=self.branch_id,
                )
            elif isinstance(payload, ToolCompleted):
                for message in payload.result.messages:
                    session.append_message(
                        event.run_id,
                        event.sequence,
                        message,
                    )
                draft = SessionRecordDraft.messages_appended(
                    session_id=self.session_id,
                    agent_id=event.agent_id,
                    sequence=event.sequence,
                    run_id=event.run_id,
                    messages=payload.result.messages,
                    branch_id=self.branch_id,
                )
            else:
                session.finish_run(
                    event.run_id,
                    event.sequence,
                    payload.result,
                )
                draft = SessionRecordDraft.run_finished(
                    session_id=self.session_id,
                    agent_id=event.agent_id,
                    sequence=event.sequence,
                    run_id=event.run_id,
                    result=payload.result,
                    branch_id=self.branch_id,
                )

            if journal_storage is not None:
                await journal_storage.append(
                    draft,
                    expected_head_id=expected_head_id,
                    check_head=True,
                )
            else:
                await self.storage.save(session)

    async def load(self) -> AgentSession | None:
        """Load the currently persisted detached Session snapshot."""

        if isinstance(self.storage, SessionJournalStorage):
            checkout = await self.storage.checkout(
                self.session_id,
                branch_id=self.branch_id,
            )
            return checkout.session if checkout is not None else None
        return await self.storage.load(self.session_id)
