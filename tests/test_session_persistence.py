import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from simagentplg import (
    SESSION_SCHEMA_VERSION,
    AgentRunResult,
    AgentSession,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    JsonlSessionStorage,
    ModelAdapter,
    RunStatus,
    RunUsage,
    SessionBranchIntent,
    SessionConflictError,
    SessionLockTimeoutError,
    SessionRecordDraft,
    SessionRecorder,
    SessionRecordKind,
    SessionSerializationError,
    SessionStorageError,
    SessionTreeStorage,
    StopReason,
    SummaryEntry,
    session_from_dict,
    session_to_dict,
)


class SequenceModel(ModelAdapter):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.contexts: list[Any] = []

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        self.contexts.append(context)
        return AssistantMessage(content=self.responses.pop(0))


def completed_result(output: str = "完成") -> AgentRunResult:
    return AgentRunResult(
        status=RunStatus.COMPLETED,
        stop_reason=StopReason.TEXT_RESPONSE,
        turns=2,
        output=output,
        usage=RunUsage(
            input_tokens=12,
            output_tokens=3,
            total_tokens=15,
            request_count=1,
            reported_request_count=1,
            cache_read_tokens=2,
            reasoning_tokens=1,
        ),
    )


def durable_session(session_id: str = "持久会话") -> AgentSession:
    session = AgentSession(session_id=session_id)
    session.bind_agent("durable-agent")
    session.begin_run("run-1", "记住 café", 1)
    session.append_message(
        "run-1",
        2,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"北京"}',
                    },
                }
            ],
        },
    )
    session.append_message(
        "run-1",
        3,
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"value":"晴天"}',
        },
    )
    session.append_message(
        "run-1",
        4,
        {
            "role": "assistant",
            "content": "完成",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
                "cache_read_tokens": 2,
                "cache_write_tokens": None,
                "reasoning_tokens": 1,
            },
        },
    )
    session.finish_run("run-1", 5, completed_result())
    summary = SummaryEntry(
        content="用户要求记住 café；北京查询结果为晴天。",
        source="summary-model:test",
        history_start_index=1,
        first_kept_index=3,
        summarized_message_count=2,
        tokens_before=15,
    )
    session.apply_compaction(
        "compact-1",
        6,
        summary,
        (
            summary.to_agent_message(),
            {"role": "assistant", "content": "完成"},
        ),
    )
    return session


def json_files(directory: str) -> list[Path]:
    return list(Path(directory).glob("*.jsonl"))


def jsonl_lock_files(directory: str) -> list[Path]:
    return list(Path(directory).glob("*.jsonl.lock"))


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def directory_entries(directory: str) -> list[Path]:
    return list(Path(directory).iterdir())


def append_bytes(path: Path, content: bytes) -> None:
    with path.open("ab") as stream:
        stream.write(content)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json_lines(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def start_append_process(
    directory: str,
    session_id: str,
    expected_head_id: str,
    run_id: str,
    branch_id: str,
    start_path: Path,
) -> subprocess.Popen[str]:
    script = """
import asyncio
import json
import sys
import time
from pathlib import Path
from simagentplg import (
    JsonlSessionStorage,
    SessionConflictError,
    SessionRecordDraft,
)

async def main():
    root, session_id, expected_head, run_id, branch_id, start_path = sys.argv[1:]
    while not Path(start_path).exists():
        time.sleep(0.005)
    storage = JsonlSessionStorage(root)
    draft = SessionRecordDraft.run_started(
        session_id=session_id,
        agent_id="durable-agent",
        sequence=1,
        run_id=run_id,
        task=run_id,
        branch_id=branch_id,
    )
    try:
        record = await storage.append(
            draft,
            expected_head_id=expected_head,
            check_head=True,
        )
    except SessionConflictError:
        result = {"status": "conflict"}
    else:
        result = {
            "status": "success",
            "record_id": record.record_id,
            "parent_id": record.parent_id,
            "revision": record.revision,
        }
    print(json.dumps(result))

asyncio.run(main())
"""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            directory,
            session_id,
            expected_head_id,
            run_id,
            branch_id,
            str(start_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


async def collect_process(process: subprocess.Popen[str]) -> dict[str, Any]:
    stdout, stderr = await asyncio.to_thread(process.communicate, timeout=10)
    if process.returncode != 0:
        raise AssertionError(f"subprocess failed: {stderr}")
    return json.loads(stdout)


def start_python_process(script: str, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", script, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


class SessionPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def test_versioned_codec_round_trips_complete_session(self) -> None:
        session = durable_session()

        payload = session_to_dict(session)
        restored = session_from_dict(payload)

        self.assertEqual(payload["schema_version"], SESSION_SCHEMA_VERSION)
        self.assertEqual(restored, session)
        self.assertEqual(restored.runs[0].result, completed_result())
        self.assertEqual(
            restored.compactions[0].summary.content,
            session.compactions[0].summary.content,
        )

    async def test_jsonl_storage_round_trips_between_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = JsonlSessionStorage(directory)
            reader = JsonlSessionStorage(directory)
            session = durable_session()

            await writer.save(session)
            restored = await reader.load(session.session_id)

            self.assertEqual(restored, session)
            files = await asyncio.to_thread(json_files, directory)
            self.assertEqual(len(files), 1)
            self.assertNotIn("持久会话", files[0].name)

    async def test_session_id_cannot_escape_storage_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sessions"
            storage = JsonlSessionStorage(root)
            session = durable_session("../../outside")

            await storage.save(session)

            self.assertEqual(len(list(root.glob("*.jsonl"))), 1)
            self.assertFalse((Path(directory) / "outside.json").exists())

    async def test_corrupt_and_unknown_schema_are_explicit_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            session = durable_session("broken")
            await storage.save(session)
            path = (await asyncio.to_thread(json_files, directory))[0]

            await asyncio.to_thread(write_text, path, "{not-json\n")
            with self.assertRaises(SessionSerializationError):
                await storage.load("broken")

            await asyncio.to_thread(
                write_text,
                path,
                json.dumps({"journal_schema_version": 999}) + "\n",
            )
            with self.assertRaisesRegex(
                SessionSerializationError,
                "unsupported.*999",
            ):
                await storage.load("broken")

    async def test_incomplete_tail_is_ignored_and_repaired_on_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            original = durable_session("partial-tail")
            await storage.save(original)
            path = (await asyncio.to_thread(json_files, directory))[0]
            await asyncio.to_thread(append_bytes, path, b'{"partial":')

            restored = await storage.load("partial-tail")
            self.assertEqual(restored, original)

            changed = original.snapshot()
            changed.begin_run("run-2", "new task", 1)
            changed.append_message(
                "run-2",
                2,
                {"role": "assistant", "content": "new answer"},
            )
            changed.finish_run("run-2", 3, completed_result("new answer"))
            await storage.save(changed)

            self.assertEqual(await storage.load("partial-tail"), changed)
            self.assertEqual(len(await storage.records("partial-tail")), 2)

    async def test_broken_tree_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            session = durable_session("broken-parent")
            await storage.save(session)
            await storage.save(session)
            path = (await asyncio.to_thread(json_files, directory))[0]
            records = await asyncio.to_thread(read_json_lines, path)
            records[1]["parent_id"] = "not-the-first-record"
            await asyncio.to_thread(write_json_lines, path, records)

            with self.assertRaisesRegex(
                SessionSerializationError,
                "parent changed",
            ):
                await storage.load("broken-parent")

    async def test_non_json_message_is_rejected_without_creating_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            session = AgentSession(session_id="invalid-json")
            session.bind_agent("durable-agent")
            session.begin_run("run-1", "task", 1)
            session.append_message(
                "run-1",
                2,
                {"role": "assistant", "content": {"not-json"}},
            )

            with self.assertRaises(SessionSerializationError):
                await storage.save(session)
            self.assertEqual(
                await asyncio.to_thread(directory_entries, directory),
                [],
            )

    async def test_failed_append_preserves_previous_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            original = durable_session("atomic")
            await storage.save(original)
            changed = original.snapshot()
            changed.begin_run("run-2", "new task", 1)
            changed.append_message(
                "run-2",
                2,
                {"role": "assistant", "content": "new answer"},
            )
            changed.finish_run("run-2", 3, completed_result("new answer"))

            with (
                patch(
                    "simagentplg.session.jsonl.os.write",
                    side_effect=OSError("disk failure"),
                ),
                self.assertRaises(SessionStorageError),
            ):
                await storage.save(changed)

            restored = await storage.load("atomic")
            self.assertEqual(restored, original)

    async def test_separate_python_process_loads_saved_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            await storage.save(durable_session("cross-process"))
            script = """
import asyncio
import json
import sys
from simagentplg import JsonlSessionStorage

async def main():
    session = await JsonlSessionStorage(sys.argv[1]).load(sys.argv[2])
    if session is None:
        raise RuntimeError("missing session")
    print(json.dumps({
        "agent_id": session.agent_id,
        "roles": [message["role"] for message in session.messages],
        "runs": len(session.runs),
    }, ensure_ascii=False))

asyncio.run(main())
"""

            process = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-c",
                    script,
                    directory,
                    "cross-process",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            loaded = json.loads(process.stdout)

            self.assertEqual(loaded["agent_id"], "durable-agent")
            self.assertEqual(loaded["roles"], ["system", "assistant"])
            self.assertEqual(loaded["runs"], 1)

    async def test_storage_implements_backend_neutral_tree_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)

            self.assertIsInstance(storage, SessionTreeStorage)

    async def test_missing_read_does_not_create_storage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sessions"
            storage = JsonlSessionStorage(root)

            self.assertIsNone(await storage.load("missing"))
            self.assertFalse(root.exists())

    async def test_competing_processes_conditionally_append_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            await storage.save(durable_session("process-race"))
            head = await storage.head("process-race")
            assert head is not None
            start_path = Path(directory) / "start"
            processes = [
                start_append_process(
                    directory,
                    "process-race",
                    head.record_id,
                    f"competing-run-{index}",
                    "main",
                    start_path,
                )
                for index in range(2)
            ]
            await asyncio.to_thread(write_text, start_path, "go")

            results = await asyncio.gather(
                *(collect_process(process) for process in processes)
            )

            self.assertEqual(
                sorted(result["status"] for result in results),
                ["conflict", "success"],
            )
            records = await storage.records("process-race")
            self.assertEqual([record.revision for record in records], [1, 2])
            self.assertEqual(records[1].parent_id, head.record_id)

    async def test_processes_append_different_branches_with_correct_parents(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            await storage.save(durable_session("branch-race"))
            main_head = await storage.head("branch-race")
            assert main_head is not None
            branch = await storage.fork(
                "branch-race",
                branch_id="experiment",
            )
            start_path = Path(directory) / "start"
            processes = [
                start_append_process(
                    directory,
                    "branch-race",
                    main_head.record_id,
                    "main-run",
                    "main",
                    start_path,
                ),
                start_append_process(
                    directory,
                    "branch-race",
                    branch.head.record_id,
                    "branch-run",
                    "experiment",
                    start_path,
                ),
            ]
            await asyncio.to_thread(write_text, start_path, "go")

            results = await asyncio.gather(
                *(collect_process(process) for process in processes)
            )

            self.assertEqual(
                [result["status"] for result in results],
                ["success", "success"],
            )
            records = await storage.records("branch-race")
            appended = {
                record.data.get("run_id"): record
                for record in records
                if record.kind is SessionRecordKind.RUN_STARTED
            }
            self.assertEqual(appended["main-run"].parent_id, main_head.record_id)
            self.assertEqual(
                appended["branch-run"].parent_id,
                branch.head.record_id,
            )
            self.assertEqual(
                [record.revision for record in records],
                list(range(1, len(records) + 1)),
            )

    async def test_lock_timeout_is_explicit_and_wait_is_cancellable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            await storage.save(durable_session("held-lock"))
            lock_path = (await asyncio.to_thread(jsonl_lock_files, directory))[0]
            release_path = Path(directory) / "release"
            script = """
import fcntl
import os
import sys
import time
from pathlib import Path

descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX)
print("locked", flush=True)
while not Path(sys.argv[2]).exists():
    time.sleep(0.005)
fcntl.flock(descriptor, fcntl.LOCK_UN)
os.close(descriptor)
"""
            holder = await asyncio.to_thread(
                start_python_process,
                script,
                str(lock_path),
                str(release_path),
            )
            assert holder.stdout is not None
            self.assertEqual(
                await asyncio.to_thread(holder.stdout.readline),
                "locked\n",
            )
            try:
                with self.assertRaises(SessionLockTimeoutError):
                    await JsonlSessionStorage(
                        directory,
                        lock_timeout=0.05,
                    ).load("held-lock")

                waiting = asyncio.create_task(
                    JsonlSessionStorage(
                        directory,
                        lock_timeout=None,
                    ).load("held-lock")
                )
                await asyncio.sleep(0.03)
                waiting.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(waiting, timeout=1)
            finally:
                await asyncio.to_thread(write_text, release_path, "release")
                _, stderr = await asyncio.to_thread(holder.communicate, timeout=10)
            self.assertEqual(holder.returncode, 0, stderr)

    async def test_crashed_lock_holder_releases_and_tail_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            original = durable_session("crashed-writer")
            await storage.save(original)
            journal_path = (await asyncio.to_thread(json_files, directory))[0]
            lock_path = (await asyncio.to_thread(jsonl_lock_files, directory))[0]
            script = """
import fcntl
import os
import sys

lock_descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
journal_descriptor = os.open(sys.argv[2], os.O_WRONLY | os.O_APPEND)
os.write(journal_descriptor, b'{"partial":')
os.fsync(journal_descriptor)
os._exit(23)
"""
            crashed = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-c",
                    script,
                    str(lock_path),
                    str(journal_path),
                ],
                check=False,
            )
            self.assertEqual(crashed.returncode, 23)
            self.assertEqual(await storage.load("crashed-writer"), original)
            head = await storage.head("crashed-writer")
            assert head is not None
            appended = await storage.append(
                SessionRecordDraft.run_started(
                    session_id="crashed-writer",
                    agent_id="durable-agent",
                    sequence=1,
                    run_id="after-crash",
                    task="continue",
                ),
                expected_head_id=head.record_id,
                check_head=True,
            )

            self.assertEqual(appended.revision, 2)
            self.assertEqual(len(await storage.records("crashed-writer")), 2)
            await asyncio.to_thread(read_json_lines, journal_path)

    async def test_restored_agent_continues_and_updates_durable_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            recorder = SessionRecorder(session_id="resume", storage=storage)
            first_agent = BaseAgent(
                SequenceModel(["saved answer"]),
                agent_id="durable-agent",
                event_sink=recorder,
            )
            await first_agent.run(task="saved task")

            saved = await JsonlSessionStorage(directory).load("resume")
            assert saved is not None
            resumed_model = SequenceModel(["continued answer"])
            resumed_agent = BaseAgent(
                resumed_model,
                agent_id="durable-agent",
                event_sink=SessionRecorder(
                    session_id="resume",
                    storage=JsonlSessionStorage(directory),
                ),
            )
            resumed_agent.restore_session(saved)

            result = await resumed_agent.run(task="continue")
            updated = await storage.load("resume")

            self.assertEqual(result.output, "continued answer")
            assert updated is not None
            self.assertEqual(len(updated.runs), 2)
            records = await storage.records("resume")
            self.assertEqual(
                [record.kind for record in records],
                [
                    SessionRecordKind.RUN_STARTED,
                    SessionRecordKind.MESSAGE_APPENDED,
                    SessionRecordKind.RUN_FINISHED,
                    SessionRecordKind.RUN_STARTED,
                    SessionRecordKind.MESSAGE_APPENDED,
                    SessionRecordKind.RUN_FINISHED,
                ],
            )
            self.assertEqual(
                [record.revision for record in records],
                list(range(1, 7)),
            )
            self.assertEqual(
                [record.parent_id for record in records[1:]],
                [record.record_id for record in records[:-1]],
            )
            self.assertTrue(all(record.branch_id == "main" for record in records))
            self.assertEqual(
                [
                    message.get("content")
                    for message in resumed_model.contexts[0].agent_messages
                ],
                [
                    resumed_agent.system_prompt,
                    "saved task",
                    "saved answer",
                    "continue",
                ],
            )

    async def test_fork_continues_without_changing_main_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            first_agent = BaseAgent(
                SequenceModel(["main answer"]),
                agent_id="durable-agent",
                event_sink=SessionRecorder(session_id="tree", storage=storage),
            )
            await first_agent.run(task="main task")
            main_head = await storage.head("tree")
            assert main_head is not None

            forked = await storage.fork("tree", branch_id="experiment")
            self.assertEqual(forked.branch.intent, SessionBranchIntent.FORK)
            self.assertEqual(forked.branch.base_record_id, main_head.record_id)
            self.assertEqual(forked.head.parent_id, main_head.record_id)
            self.assertEqual(len(forked.session.runs), 1)

            branch_agent = BaseAgent(
                SequenceModel(["branch answer"]),
                agent_id="durable-agent",
                event_sink=SessionRecorder(
                    session_id="tree",
                    storage=storage,
                    branch_id="experiment",
                ),
            )
            branch_agent.restore_session(forked.session)
            await branch_agent.run(task="branch task")

            main = await storage.load("tree")
            experiment = await storage.checkout("tree", branch_id="experiment")
            assert main is not None
            assert experiment is not None
            self.assertEqual([run.task for run in main.runs], ["main task"])
            self.assertEqual(
                [run.task for run in experiment.session.runs],
                ["main task", "branch task"],
            )
            branches = await storage.list_branches("tree")
            self.assertEqual(
                [branch.branch_id for branch in branches],
                ["main", "experiment"],
            )
            records = await storage.records("tree")
            self.assertEqual(
                [record.revision for record in records],
                list(range(1, len(records) + 1)),
            )

    async def test_rollback_requires_ancestor_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            recorder = SessionRecorder(session_id="rollback", storage=storage)
            agent = BaseAgent(
                SequenceModel(["one", "two"]),
                agent_id="durable-agent",
                event_sink=recorder,
            )
            await agent.run(task="first")
            first_finish = (await storage.records("rollback"))[-1]
            await agent.run(task="second")

            rolled_back = await storage.rollback(
                "rollback",
                to_record_id=first_finish.record_id,
                branch_id="before-second",
            )
            self.assertEqual(rolled_back.branch.intent, SessionBranchIntent.ROLLBACK)
            self.assertEqual([run.task for run in rolled_back.session.runs], ["first"])
            main = await storage.load("rollback")
            assert main is not None
            self.assertEqual([run.task for run in main.runs], ["first", "second"])

            unrelated = await storage.fork(
                "rollback",
                branch_id="unrelated",
            )
            with self.assertRaisesRegex(ValueError, "not an ancestor"):
                await storage.rollback(
                    "rollback",
                    to_record_id=unrelated.head.record_id,
                    source_branch="main",
                    branch_id="invalid-rollback",
                )

    async def test_prepare_retry_reuses_task_from_before_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            agent = BaseAgent(
                SequenceModel(["first result", "original result"]),
                agent_id="durable-agent",
                event_sink=SessionRecorder(session_id="retry", storage=storage),
            )
            await agent.run(task="first task")
            await agent.run(task="repeat me")
            main = await storage.load("retry")
            assert main is not None
            retried_run_id = main.runs[-1].run_id

            retry = await storage.prepare_retry(
                "retry",
                run_id=retried_run_id,
                branch_id="retry-second",
            )
            self.assertEqual(retry.task, "repeat me")
            self.assertEqual(retry.checkout.branch.intent, SessionBranchIntent.RETRY)
            self.assertEqual(
                [run.task for run in retry.checkout.session.runs],
                ["first task"],
            )

            retry_agent = BaseAgent(
                SequenceModel(["new result"]),
                agent_id="durable-agent",
                event_sink=SessionRecorder(
                    session_id="retry",
                    storage=storage,
                    branch_id="retry-second",
                ),
            )
            retry_agent.restore_session(retry.checkout.session)
            await retry_agent.run(task=retry.task)

            retry_checkout = await storage.checkout(
                "retry",
                branch_id="retry-second",
            )
            assert retry_checkout is not None
            self.assertEqual(
                [run.result.output for run in retry_checkout.session.runs],
                ["first result", "new result"],
            )
            unchanged_main = await storage.load("retry")
            assert unchanged_main is not None
            self.assertEqual(unchanged_main.runs[-1].result.output, "original result")

    async def test_first_run_retry_starts_from_bound_empty_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            agent = BaseAgent(
                SequenceModel(["original"]),
                agent_id="durable-agent",
                event_sink=SessionRecorder(session_id="retry-first", storage=storage),
            )
            await agent.run(task="first task")
            main = await storage.load("retry-first")
            assert main is not None

            retry = await storage.prepare_retry(
                "retry-first",
                run_id=main.runs[0].run_id,
                branch_id="retry-root",
            )

            self.assertEqual(retry.task, "first task")
            self.assertEqual(retry.checkout.session.agent_id, "durable-agent")
            self.assertEqual(retry.checkout.session.runs, [])
            self.assertIsNone(retry.checkout.head.parent_id)

    async def test_conditional_append_rejects_stale_branch_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            session = durable_session("conflict")
            await storage.save(session)
            head = await storage.head("conflict")
            assert head is not None
            draft = SessionRecordDraft.run_started(
                session_id="conflict",
                agent_id="durable-agent",
                sequence=1,
                run_id="run-2",
                task="new task",
            )

            with self.assertRaises(SessionConflictError):
                await storage.append(
                    draft,
                    expected_head_id="stale-head",
                    check_head=True,
                )
            self.assertEqual((await storage.head("conflict")), head)

    async def test_checkout_allows_audit_of_unfinished_node_but_fork_rejects_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonlSessionStorage(directory)
            agent = BaseAgent(
                SequenceModel(["answer"]),
                agent_id="durable-agent",
                event_sink=SessionRecorder(session_id="audit-node", storage=storage),
            )
            await agent.run(task="task")
            started = (await storage.records("audit-node"))[0]

            audited = await storage.checkout(
                "audit-node",
                record_id=started.record_id,
            )
            assert audited is not None
            self.assertFalse(audited.session.runs[0].finished)
            with self.assertRaisesRegex(ValueError, "unfinished run"):
                await storage.fork(
                    "audit-node",
                    from_record_id=started.record_id,
                    branch_id="unsafe",
                )

    def test_restore_rejects_wrong_agent_and_unfinished_run(self) -> None:
        wrong_agent = BaseAgent(
            SequenceModel(["unused"]),
            agent_id="other-agent",
        )
        with self.assertRaisesRegex(ValueError, "belongs to agent"):
            wrong_agent.restore_session(durable_session())

        unfinished = AgentSession(session_id="unfinished")
        unfinished.bind_agent("durable-agent")
        unfinished.begin_run("run-open", "task", 1)
        matching_agent = BaseAgent(
            SequenceModel(["unused"]),
            agent_id="durable-agent",
        )
        with self.assertRaisesRegex(ValueError, "unfinished"):
            matching_agent.restore_session(unfinished)


if __name__ == "__main__":
    unittest.main()
