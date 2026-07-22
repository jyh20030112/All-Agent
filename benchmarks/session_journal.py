"""Measure JSONL Session journal append and replay scaling."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from ejagent import AgentSession, JsonlSessionStorage


def journal_size(directory: str) -> int:
    return next(Path(directory).glob("*.jsonl")).stat().st_size


async def benchmark(record_count: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        storage = JsonlSessionStorage(directory)
        session = AgentSession(session_id=f"benchmark-{record_count}")
        session.bind_agent("benchmark-agent")

        started = time.perf_counter()
        for _ in range(record_count):
            await storage.save(session)
        build_seconds = time.perf_counter() - started

        started = time.perf_counter()
        await JsonlSessionStorage(directory).load(session.session_id)
        load_seconds = time.perf_counter() - started

        byte_count = await asyncio.to_thread(journal_size, directory)
        return {
            "records": record_count,
            "bytes": byte_count,
            "build_seconds": round(build_seconds, 6),
            "average_append_ms": round(
                build_seconds * 1000 / record_count,
                6,
            ),
            "load_ms": round(load_seconds * 1000, 6),
        }


async def main(record_counts: list[int]) -> None:
    for record_count in record_counts:
        print(json.dumps(await benchmark(record_count), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "records",
        type=int,
        nargs="*",
        default=[100, 500, 1000],
    )
    args = parser.parse_args()
    if any(record_count <= 0 for record_count in args.records):
        parser.error("record counts must be greater than zero")
    asyncio.run(main(args.records))
