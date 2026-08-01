from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

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
    RunUsage,
    SessionCommit,
    SessionConflictError,
    SessionMigrationError,
    SessionSnapshot,
    SessionStore,
    SessionStoreSerializationError,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ejagent.harness import MemorySessionStore
from ejagent.storage import JsonlSessionStore
from ejagent.storage._legacy import (
    LegacyRun,
    LegacyRunResult,
    LegacySessionData,
)
from ejagent.storage.codec import (
    session_commit_from_dict,
    session_commit_to_dict,
)
from ejagent.storage.migration import migrate_legacy_session

_NOW = datetime(2026, 8, 1, 8, 30, tzinfo=UTC)


def _legacy_result(*, output: str | None = None) -> LegacyRunResult:
    return LegacyRunResult(
        status=RunStatus.COMPLETED,
        stop_reason=StopReason.TEXT_RESPONSE,
        turns=2,
        output=output,
        error=None,
        usage=RunUsage(),
    )


def _write_legacy_checkpoint(
    root: Path,
    *,
    session_id: str,
    agent_id: str,
    messages: list[dict[str, Any]],
    run_id: str,
    output: str | None,
) -> None:
    usage = RunUsage().to_dict()
    document = {
        "schema_version": 1,
        "session": {
            "session_id": session_id,
            "agent_id": agent_id,
            "entries": [
                {"run_id": run_id, "sequence": index + 1, "message": message}
                for index, message in enumerate(messages)
            ],
            "runs": [
                {
                    "run_id": run_id,
                    "task": messages[0]["content"],
                    "intent": "task",
                    "start_sequence": 1,
                    "finish_sequence": len(messages) + 1,
                    "result": {
                        "status": "completed",
                        "stop_reason": "text_response",
                        "turns": 2,
                        "output": output,
                        "error": None,
                        "usage": usage,
                    },
                }
            ],
            "compactions": [],
        },
    }
    record = {
        "journal_schema_version": 1,
        "record_id": "legacy-checkpoint",
        "parent_id": None,
        "branch_id": "main",
        "revision": 1,
        "session_id": session_id,
        "agent_id": agent_id,
        "sequence": 0,
        "type": "checkpoint",
        "data": {"document": document},
    }
    digest = sha256(session_id.encode("utf-8")).hexdigest()
    (root / f"{digest}.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_legacy_event_journal(root: Path) -> None:
    usage = RunUsage().to_dict()
    drafts = (
        ("run_started", 1, {"run_id": "event-run", "task": "remember blue"}),
        (
            "message_appended",
            2,
            {
                "run_id": "event-run",
                "message": {"role": "assistant", "content": "noted"},
            },
        ),
        (
            "run_finished",
            3,
            {
                "run_id": "event-run",
                "result": {
                    "status": "completed",
                    "stop_reason": "text_response",
                    "turns": 1,
                    "output": "noted",
                    "error": None,
                    "usage": usage,
                },
            },
        ),
    )
    records: list[dict[str, Any]] = []
    parent_id: str | None = None
    for revision, (kind, sequence, data) in enumerate(drafts, start=1):
        record_id = f"record-{revision}"
        records.append(
            {
                "journal_schema_version": 1,
                "record_id": record_id,
                "parent_id": parent_id,
                "branch_id": "main",
                "revision": revision,
                "session_id": "event-session",
                "agent_id": "event-agent",
                "sequence": sequence,
                "type": kind,
                "data": data,
            }
        )
        parent_id = record_id
    digest = sha256(b"event-session").hexdigest()
    (root / f"{digest}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


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


class MemorySessionStoreContractTests(
    _SessionStoreContract,
    unittest.IsolatedAsyncioTestCase,
):
    store_factory = MemorySessionStore


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

    async def test_auto_migrates_legacy_jsonl_once(self) -> None:
        _write_legacy_checkpoint(
            self.root,
            session_id="legacy-session",
            agent_id="legacy-agent",
            run_id="legacy-run",
            output="已记住",
            messages=[
                {"role": "user", "content": "记住 café"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "remember",
                                "arguments": '{"text":"你好"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
                {"role": "assistant", "content": "已记住"},
            ],
        )

        migrating = JsonlSessionStore(
            self.root,
            legacy_session_id="legacy-session",
        )
        snapshot = await migrating.load("legacy-agent")

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.messages[0], UserMessage("记住 café"))
        self.assertEqual(
            snapshot.messages[1],
            AssistantMessage(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="remember",
                        arguments={"text": "你好"},
                    ),
                )
            ),
        )
        self.assertEqual(
            snapshot.messages[2],
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="remember",
                result="ok",
            ),
        )
        self.assertEqual(len(await migrating.load_audit("legacy-agent")), 1)
        self.assertEqual(
            await JsonlSessionStore(self.root).load("legacy-agent"),
            snapshot,
        )
        self.assertEqual(len(tuple(self.root.glob("*.core.jsonl"))), 1)

    async def test_migrates_event_journal_without_legacy_runtime(self) -> None:
        _write_legacy_event_journal(self.root)
        store = JsonlSessionStore(
            self.root,
            legacy_session_id="event-session",
        )

        snapshot = await store.load("event-agent")

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(
            snapshot.messages,
            (UserMessage("remember blue"), AssistantMessage("noted")),
        )
        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.last_result.output, "noted")  # type: ignore[union-attr]

    async def test_migration_errors_are_actionable(self) -> None:
        unfinished = LegacySessionData(
            session_id="unfinished",
            agent_id="agent-1",
            entries=({"role": "user", "content": "work"},),
            runs=(LegacyRun("run-open"),),
        )

        with self.assertRaises(SessionMigrationError) as raised:
            migrate_legacy_session(unfinished)

        self.assertIn("unfinished", str(raised.exception))
        self.assertIn("Remediation:", str(raised.exception))
        self.assertTrue(raised.exception.remediation)

        unsupported = LegacySessionData(
            session_id="unsupported",
            agent_id="agent-1",
            entries=(
                {"role": "user", "content": "work"},
                {"role": "developer", "content": "hidden"},
            ),
            runs=(LegacyRun("run-1", _legacy_result()),),
        )
        with self.assertRaises(SessionMigrationError) as unsupported_error:
            migrate_legacy_session(unsupported)
        self.assertIn("unsupported", str(unsupported_error.exception))
