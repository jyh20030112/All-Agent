"""Inspect and create durable Session branches without replaying a model."""

import argparse
import asyncio
import os
from pathlib import Path

from ejagent import JsonlSessionStorage


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default="durable-session-demo")
    parser.add_argument(
        "--session-dir",
        default=os.getenv("EJAGENT_SESSION_DIR", ".ejagent-sessions"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("branches")

    fork_parser = commands.add_parser("fork")
    fork_parser.add_argument("branch_id")
    fork_parser.add_argument("--from-record-id")

    rollback_parser = commands.add_parser("rollback")
    rollback_parser.add_argument("to_record_id")
    rollback_parser.add_argument("branch_id")

    retry_parser = commands.add_parser("retry")
    retry_parser.add_argument("run_id")
    retry_parser.add_argument("branch_id")

    args = parser.parse_args()
    storage = JsonlSessionStorage(Path(args.session_dir))
    if args.command == "branches":
        for branch in await storage.list_branches(args.session_id):
            print(
                f"{branch.branch_id}: head={branch.head_record_id} "
                f"intent={branch.intent or 'main'}"
            )
    elif args.command == "fork":
        checkout = await storage.fork(
            args.session_id,
            from_record_id=args.from_record_id,
            branch_id=args.branch_id,
        )
        print(f"created {checkout.branch.branch_id} at {checkout.head.record_id}")
    elif args.command == "rollback":
        checkout = await storage.rollback(
            args.session_id,
            to_record_id=args.to_record_id,
            branch_id=args.branch_id,
        )
        print(f"created {checkout.branch.branch_id} at {checkout.head.record_id}")
    else:
        retry = await storage.prepare_retry(
            args.session_id,
            run_id=args.run_id,
            branch_id=args.branch_id,
        )
        print(f"prepared {retry.checkout.branch.branch_id}; rerun task: {retry.task}")


if __name__ == "__main__":
    asyncio.run(main())
