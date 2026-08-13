"""End-to-end AgentHarness guide: custom tools, runtime controls, durable store.

Three subcommands walk through the Harness extension points:

  record    -- one Run that calls a custom tool, committed durably.
  resume    -- a brand-new Harness over the same store recovers the
               committed Conversation (including the tool result message).
  controls  -- live steer(), follow_up(), and cancel() against a running
               agent, plus an observer that prints each RunAudit summary.

Requires the usual Provider env vars (see .env.example). Durable state is
written to --session-dir (default .ejagent-sessions) as append-only JSONL.
"""

import argparse
import asyncio
import os
from pathlib import Path

from ejagent.contracts import (
    CancellationToken,
    RunAudit,
    RunObserver,
    SystemMessage,
    ToolCall,
    ToolControl,
    ToolDefinition,
    ToolExecutionResult,
    ToolSemantics,
)
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.storage import JsonlSessionStore
from ejagent.tools import FunctionTool, FunctionToolExecutor

AGENT_ID = "harness-guide"
SYSTEM_PROMPT = (
    "You are a precise assistant. Use the provided tools when asked, "
    "answer from the conversation, and keep replies short."
)

# --- Custom tool 1: a deterministic in-process function tool ---------------

ADD_TOOL = ToolDefinition(
    name="add",
    description="Add two numbers and return their sum.",
    input_schema={
        "type": "object",
        "properties": {
            "left": {"type": "number"},
            "right": {"type": "number"},
        },
        # JsonValue's static array type is a tuple; literals stay list-shaped
        # because freeze_json_value() normalizes them at runtime.
        "required": ["left", "right"],  # type: ignore[dict-item]
    },
    # Declared read-only + idempotent, so Core may safely retry/reorder it.
    semantics=ToolSemantics.read_only(),
)


async def add(call: ToolCall, cancellation: CancellationToken) -> ToolExecutionResult:
    cancellation.raise_if_cancelled()
    left = call.arguments.get("left")
    right = call.arguments.get("right")
    if (
        isinstance(left, bool)
        or not isinstance(left, (int, float))
        or isinstance(right, bool)
        or not isinstance(right, (int, float))
    ):
        return ToolExecutionResult(
            {"status": "error", "error": "left and right must be numbers"},
            error="left and right must be numbers",
        )
    return ToolExecutionResult(
        {"status": "success", "value": left + right},
        control=ToolControl.COMPLETE,  # finish the Run explicitly
        output=str(left + right),
    )


# --- Custom tool 2: a sleeping tool that gives controls a mid-flight window --

PAUSE_TOOL = ToolDefinition(
    name="pause",
    description=(
        "Pause execution for the requested number of seconds (max 10). "
        "Useful to create a short delay before the final answer."
    ),
    input_schema={
        "type": "object",
        "properties": {"seconds": {"type": "number"}},
        "required": ["seconds"],  # type: ignore[dict-item]
    },
    semantics=ToolSemantics.read_only(),
)


async def pause(
    call: ToolCall,
    cancellation: CancellationToken,
) -> ToolExecutionResult:
    cancellation.raise_if_cancelled()
    raw = call.arguments.get("seconds")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return ToolExecutionResult(
            {"status": "error", "error": "seconds must be a number"},
            error="seconds must be a number",
        )
    seconds = min(max(float(raw), 0.0), 10.0)
    await asyncio.sleep(seconds)
    return ToolExecutionResult({"status": "ok", "slept": seconds})


# --- Observer: runs after the Store decision, never alters execution ---------


class PrintObserver(RunObserver):
    """Print a one-line summary per finished Run (delivered after commit)."""

    async def observe(self, audit: RunAudit) -> None:
        state = "committed" if audit.committed else "not-committed"
        print(
            f"  [observer] run {audit.run_id[:8]} → "
            f"{audit.result.status.value} ({state}, "
            f"revision {audit.base_revision}→{audit.resulting_revision}, "
            f"{len(audit.records)} audit records)"
        )


def make_harness(
    store: JsonlSessionStore,
    *,
    tools: tuple[FunctionTool, ...],
) -> AgentHarness:
    return AgentHarness(
        agent_id=AGENT_ID,
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=FunctionToolExecutor(tools),
        store=store,
        observers=(PrintObserver(),),
        initial_messages=(SystemMessage(SYSTEM_PROMPT),),
    )


async def cmd_record(store: JsonlSessionStore) -> None:
    """Run one task through a custom tool and commit it durably."""
    harness = make_harness(store, tools=(FunctionTool(ADD_TOOL, add),))
    async with harness:
        outcome = await harness.run(
            "Use the add tool to compute 19.5 + 22.5, "
            "then reply with only the resulting number."
        )
        print(f"answer: {outcome.result.output}")
    print(f"committed revision: {harness.revision}")


async def cmd_resume(store: JsonlSessionStore) -> None:
    """Recover the durable Conversation in a new Harness instance."""
    if await store.load(AGENT_ID) is None:
        raise RuntimeError("run `record` before `resume`")
    harness = make_harness(store, tools=(FunctionTool(ADD_TOOL, add),))
    async with harness:
        outcome = await harness.run(
            "Earlier you used the add tool to compute a value. "
            "Return only that exact number from the conversation."
        )
        print(f"recalled answer: {outcome.result.output}")
    print(f"audited runs: {len(await store.load_audit(AGENT_ID))}")


async def cmd_controls(store: JsonlSessionStore) -> None:
    """Drive steer(), follow_up(), and cancel() against a live Harness."""
    harness = make_harness(store, tools=(FunctionTool(PAUSE_TOOL, pause),))
    async with harness:
        # 1) steer() before any Run is admitted → NOT_RUNNING
        receipt = harness.steer("this is not admitted while idle")
        print(f"steer while idle → {receipt.status.value}")

        # 2) run in the background so controls can be injected mid-flight.
        #    The first turn uses the pause tool, guaranteeing the agent is
        #    still RUNNING when we steer and queue a follow-up below.
        main_task = asyncio.create_task(
            harness.run(
                "Call the pause tool with seconds=1, "
                "then reply with the word READY and nothing else."
            )
        )
        await asyncio.sleep(0.5)  # let the Run reach RUNNING and start the model call

        receipt = harness.steer("Before your final answer, also append the word DONE.")
        print(
            f"steer during run → {receipt.status.value} "
            f"(input {receipt.input_id[:8]})"
        )

        follow_handle = harness.follow_up("Reply with exactly the number 7.")
        print(f"follow_up admission → {follow_handle.receipt.status.value}")

        main_outcome = await main_task
        print(
            f"main run → {main_outcome.result.status.value} "
            f"({main_outcome.result.stop_reason.value}), "
            f"output: {main_outcome.result.output!r}"
        )

        follow_outcome = await follow_handle.wait()
        print(
            f"follow-up run → {follow_outcome.result.status.value}, "
            f"output: {follow_outcome.result.output!r}"
        )

        # 3) cancel a deliberately long Run; the token interrupts the tool.
        long_task = asyncio.create_task(
            harness.run("Call the pause tool with seconds=10, then reply ONLY DONE.")
        )
        await asyncio.sleep(0.5)
        changed = harness.cancel("guide demo timeout")
        print(f"cancel() returned {changed}")
        cancelled = await long_task
        print(
            f"cancelled run → {cancelled.result.status.value} "
            f"({cancelled.result.stop_reason.value})"
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("record", "resume", "controls"))
    parser.add_argument(
        "--session-dir",
        default=os.getenv("EJAGENT_SESSION_DIR", ".ejagent-sessions"),
    )
    args = parser.parse_args()

    store = JsonlSessionStore(Path(args.session_dir))
    if args.command == "record":
        await cmd_record(store)
    elif args.command == "resume":
        await cmd_resume(store)
    else:
        await cmd_controls(store)


if __name__ == "__main__":
    asyncio.run(main())
