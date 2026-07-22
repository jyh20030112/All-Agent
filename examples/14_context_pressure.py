"""Evaluate context pressure and prepare compaction with a real Provider."""

import asyncio
import os
from collections.abc import Mapping
from typing import Any

from ejagent import (
    AgentEvent,
    BaseAgent,
    CancellationToken,
    CompactionPolicy,
    ContextBudget,
    ContextPressureEvaluated,
    MethodToolHandler,
    ModelConfig,
    OpenAIModelAdapter,
    StepOutcome,
)

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_demo_file",
        "description": "Read one synthetic file used by the context demo.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


class DemoTools(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((READ_TOOL,))

    async def do_read_demo_file(
        self,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome(
            {
                "path": arguments["path"],
                "content": "A fresh synthetic file has no warnings.",
            }
        )


OLD_TOOL_OUTPUT = "\n".join(
    f"diagnostic line {index}: legacy warning" for index in range(80)
)

HISTORY = [
    {
        "role": "user",
        "content": "Inspect the old synthetic report.",
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "old-call-1",
                "type": "function",
                "function": {
                    "name": "read_demo_file",
                    "arguments": '{"path":"legacy-report.txt"}',
                },
            }
        ],
        "usage": {
            "input_tokens": 180,
            "output_tokens": 20,
            "total_tokens": 200,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "reasoning_tokens": None,
        },
    },
    {
        "role": "tool",
        "tool_call_id": "old-call-1",
        "content": OLD_TOOL_OUTPUT,
    },
    {
        "role": "assistant",
        "content": "The legacy report contains repeated warnings.",
        "usage": {
            "input_tokens": 900,
            "output_tokens": 18,
            "total_tokens": 918,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "reasoning_tokens": None,
        },
    },
    {
        "role": "user",
        "content": "Remember that finding for the next check.",
    },
    {
        "role": "assistant",
        "content": "I will retain the legacy finding as context.",
        "usage": {
            "input_tokens": 950,
            "output_tokens": 16,
            "total_tokens": 966,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "reasoning_tokens": None,
        },
    },
]


class ContextConsoleSink:
    async def emit(self, event: AgentEvent) -> None:
        payload = event.payload
        if not isinstance(payload, ContextPressureEvaluated):
            return

        estimate = payload.decision.estimate
        print(
            f"turn {payload.turn} context: total={estimate.total_tokens}, "
            f"reported={estimate.reported_tokens}, "
            f"trailing={estimate.trailing_tokens}, "
            f"heuristic={estimate.heuristic_tokens}, "
            f"source={estimate.source}"
        )
        print(
            f"threshold={payload.decision.threshold_tokens}, "
            f"should_compact={payload.decision.should_compact}"
        )
        if payload.preparation is not None:
            preparation = payload.preparation
            roles = [
                message.get("role") for message in preparation.messages_to_summarize
            ]
            print(
                f"preparation: can_compact={preparation.can_compact}, "
                f"first_kept_index={preparation.first_kept_index}, "
                f"summarize_roles={roles}"
            )


async def main() -> None:
    budget = ContextBudget(
        context_window=int(os.getenv("HARNESS_CONTEXT_WINDOW", "600")),
        reserve_tokens=int(os.getenv("HARNESS_CONTEXT_RESERVE", "100")),
        keep_recent_tokens=int(os.getenv("HARNESS_KEEP_RECENT_TOKENS", "40")),
    )
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="context-pressure-demo",
        system_prompt=(
            "This is a context-pressure demonstration. Answer the latest "
            "question in one short sentence without calling a tool."
        ),
        handlers=[DemoTools()],
        compaction_policy=CompactionPolicy(budget),
        event_sink=ContextConsoleSink(),
    )
    agent.reset(history=HISTORY)

    try:
        result = await agent.run(task="Does the old report contain repeated warnings?")
    finally:
        await agent.shutdown()

    print(f"finished: {result.status} / {result.stop_reason}")
    print(f"output: {result.output}")
    print("history was evaluated but was not compacted or mutated")


if __name__ == "__main__":
    asyncio.run(main())
