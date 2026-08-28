from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from ejagent.contracts import (
    AssistantMessage,
    AuditReader,
    AuditRecord,
    ConversationSnapshot,
    FailureCode,
    RunDelta,
    RunFailure,
    RunOutcome,
    RunPhase,
    RunResult,
    RunStatus,
    SessionCommit,
    SessionConflictError,
    SessionSnapshot,
    SessionStore,
    SessionStoreSerializationError,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ejagent.storage import JsonlSessionStore
from ejagent.storage.codec import (
    session_commit_from_dict,
    session_commit_to_dict,
)

_NOW = datetime(2026, 8, 1, 8, 30, tzinfo=UTC)


class _ContractStore(SessionStore, AuditReader):
    pass


def _success(
    run_id: str,
    *,
    agent_id: str = "agent-1",
    base: ConversationSnapshot | None = None,
    output: str = "done",
) -> SessionCommit:
    current = base or ConversationSnapshot()
    result = RunResult(
        run_id=run_id,
        status=RunStatus.COMPLETED,
        stop_reason=StopReason.TEXT_RESPONSE,
        turns=1,
        output=output,
    )
    return SessionCommit(
        agent_id=agent_id,
        base=current,
        outcome=RunOutcome(
            result=result,
            delta=RunDelta(
                base_revision=current.revision,
                messages=(UserMessage(f"task:{run_id}"), AssistantMessage(output)),
            ),
            audit_records=(
                AuditRecord(
                    run_id=run_id,
                    sequence=1,
                    kind="model.completed",
                    occurred_at=_NOW,
                    payload={"unicode": "你好"},
                ),
            ),
        ),
    )


def _failure(
    run_id: str,
    *,
    agent_id: str = "agent-1",
    base: ConversationSnapshot | None = None,
) -> SessionCommit:
    current = base or ConversationSnapshot()
    failure = RunFailure(
        phase=RunPhase.MODEL,
        code=FailureCode.RATE_LIMIT,
        message="slow down",
        retryable=True,
    )
    return SessionCommit(
        agent_id=agent_id,
        base=current,
        outcome=RunOutcome(
            result=RunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                stop_reason=StopReason.RUNTIME_ERROR,
                turns=0,
            ),
            delta=RunDelta(base_revision=current.revision),
            failure=failure,
        ),
    )


class _SessionStoreContract:
    store_factory: Callable[[], _ContractStore]
    store: _ContractStore

    async def asyncSetUp(self) -> None:
        self.store = self.store_factory()

    async def test_commit_load_and_audit(self) -> None:
        commit = _success("run-1")

        snapshot = await self.store.commit(commit)

        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.messages, commit.resulting_conversation.messages)
        self.assertEqual(snapshot.last_result, commit.outcome.result)
        self.assertEqual(await self.store.load("agent-1"), snapshot)
        self.assertEqual(await self.store.load_audit("agent-1"), (commit.audit,))

    async def test_failed_run_is_audited_without_advancing(self) -> None:
        commit = _failure("run-failed")

        snapshot = await self.store.commit(commit)

        self.assertEqual(snapshot.revision, 0)
        self.assertEqual(snapshot.messages, ())
        self.assertIsNone(snapshot.last_result)
        self.assertEqual(await self.store.load_audit("agent-1"), (commit.audit,))

    async def test_repeating_same_commit_is_idempotent(self) -> None:
        commit = _success("run-idempotent")

        first = await self.store.commit(commit)
        second = await self.store.commit(commit)

        self.assertEqual(second, first)
        self.assertEqual(len(await self.store.load_audit("agent-1")), 1)

    async def test_reusing_run_id_for_different_commit_conflicts(self) -> None:
        await self.store.commit(_success("run-reused", output="first"))

        with self.assertRaises(SessionConflictError):
            await self.store.commit(_success("run-reused", output="different"))

    async def test_only_one_stale_base_commit_wins(self) -> None:
        commits = (_success("run-a"), _success("run-b"))

        results = await asyncio.gather(
            *(self.store.commit(commit) for commit in commits),
            return_exceptions=True,
        )

        self.assertEqual(sum(isinstance(item, SessionSnapshot) for item in results), 1)
        self.assertEqual(
            sum(isinstance(item, SessionConflictError) for item in results),
            1,
        )


class JsonlSessionStoreContractTests(
    _SessionStoreContract,
    unittest.IsolatedAsyncioTestCase,
):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.store_factory = lambda: JsonlSessionStore(root)


class JsonlSessionStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    async def test_round_trips_rich_typed_commit_across_instances(self) -> None:
        base = ConversationSnapshot(
            messages=(SystemMessage("answer briefly"),),
        )
        result = RunResult(
            run_id="rich-run",
            status=RunStatus.COMPLETED,
            stop_reason=StopReason.TOOL_COMPLETION,
            turns=2,
            output="晴天",
        )
        commit = SessionCommit(
            agent_id="rich-agent",
            base=base,
            outcome=RunOutcome(
                result=result,
                delta=RunDelta(
                    base_revision=0,
                    messages=(
                        AssistantMessage(
                            tool_calls=(
                                ToolCall(
                                    id="call-1",
                                    name="weather",
                                    arguments={"city": "杭州"},
                                ),
                            )
                        ),
                        ToolResultMessage(
                            tool_call_id="call-1",
                            tool_name="weather",
                            result={"temperature": 31},
                        ),
                        AssistantMessage("晴天"),
                    ),
                ),
            ),
        )
        self.assertEqual(
            session_commit_from_dict(session_commit_to_dict(commit)),
            commit,
        )

        expected = await JsonlSessionStore(self.root).commit(commit)

        restored = await JsonlSessionStore(self.root).load("rich-agent")
        self.assertEqual(restored, expected)

    async def test_partial_tail_is_ignored_then_truncated_on_append(self) -> None:
        store = JsonlSessionStore(self.root)
        first = await store.commit(_success("run-1"))
        path = next(self.root.glob("*.core.jsonl"))
        with path.open("ab") as stream:
            stream.write(b'{"partial":')

        restarted = JsonlSessionStore(self.root)
        self.assertEqual(await restarted.load("agent-1"), first)
        await restarted.commit(
            _success("run-2", base=first.conversation, output="again")
        )

        for line in path.read_text(encoding="utf-8").splitlines():
            self.assertIsInstance(json.loads(line), dict)
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    async def test_partial_first_record_is_replaced_by_first_commit(self) -> None:
        digest = sha256(b"agent-1").hexdigest()
        path = self.root / f"{digest}.core.jsonl"
        path.write_bytes(b'{"partial":')

        snapshot = await JsonlSessionStore(self.root).commit(_success("run-1"))

        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
        self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    async def test_complete_corrupt_record_is_rejected(self) -> None:
        store = JsonlSessionStore(self.root)
        await store.commit(_success("run-1"))
        path = next(self.root.glob("*.core.jsonl"))
        with path.open("ab") as stream:
            stream.write(b"not-json\n")

        with self.assertRaises(SessionStoreSerializationError):
            await JsonlSessionStore(self.root).load("agent-1")

    async def test_separate_instances_compare_and_commit_atomically(self) -> None:
        first_store = JsonlSessionStore(self.root)
        second_store = JsonlSessionStore(self.root)
        self.assertIsNone(await first_store.load("agent-1"))
        self.assertIsNone(await second_store.load("agent-1"))

        results = await asyncio.gather(
            first_store.commit(_success("run-a")),
            second_store.commit(_success("run-b")),
            return_exceptions=True,
        )

        self.assertEqual(sum(isinstance(item, SessionSnapshot) for item in results), 1)
        self.assertEqual(
            sum(isinstance(item, SessionConflictError) for item in results),
            1,
        )
