"""Measure append and replay scaling for the Core JSONL SessionStore."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from ejagent.contracts import (
    AssistantMessage,
    ConversationSnapshot,
    RunDelta,
    RunOutcome,
    RunResult,
    RunStatus,
    SessionCommit,
    StopReason,
    UserMessage,
)
from ejagent.storage import JsonlSessionStore

AGENT_ID = "benchmark-agent"


def journal_size(directory: str) -> int:
    return next(Path(directory).glob("*.core.jsonl")).stat().st_size


def commit(index: int, base: ConversationSnapshot) -> SessionCommit:
    run_id = f"benchmark-{index}"
    return SessionCommit(
        agent_id=AGENT_ID,
        base=base,
        outcome=RunOutcome(
            result=RunResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                stop_reason=StopReason.TEXT_RESPONSE,
                turns=1,
                output="ok",
            ),
            delta=RunDelta(
                base_revision=base.revision,
                messages=(UserMessage(run_id), AssistantMessage("ok")),
            ),
        ),
    )


async def benchmark(record_count: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        store = JsonlSessionStore(directory)
        conversation = ConversationSnapshot()
        started = time.perf_counter()
        for index in range(record_count):
            snapshot = await store.commit(commit(index, conversation))
            conversation = snapshot.conversation
        build_seconds = time.perf_counter() - started

        started = time.perf_counter()
        await JsonlSessionStore(directory).load(AGENT_ID)
        load_seconds = time.perf_counter() - started
        byte_count = await asyncio.to_thread(journal_size, directory)
        return {
            "records": record_count,
            "bytes": byte_count,
            "build_seconds": round(build_seconds, 6),
            "average_append_ms": round(build_seconds * 1000 / record_count, 6),
            "load_ms": round(load_seconds * 1000, 6),
        }


async def main(record_counts: list[int]) -> None:
    for record_count in record_counts:
        print(json.dumps(await benchmark(record_count), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=int, nargs="*", default=[100, 500, 1000])
    args = parser.parse_args()
    if any(record_count <= 0 for record_count in args.records):
        parser.error("record counts must be greater than zero")
    asyncio.run(main(args.records))
