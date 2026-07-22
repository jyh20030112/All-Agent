"""Observe real Provider usage and stop before an over-budget follow-up."""

import asyncio
import os
from collections.abc import Mapping
from typing import Any

from ejagent import (
    AgentEvent,
    AgentFinished,
    BaseAgent,
    CancellationToken,
    MessageCompleted,
    MethodToolHandler,
    ModelConfig,
    OpenAIModelAdapter,
    RuntimePolicy,
    StepOutcome,
)

INSPECT_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_usage_demo",
        "description": (
            "Inspect the usage-budget demo once. Return the observation to "
            "the model so it would normally need another turn."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


class UsageTools(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((INSPECT_TOOL,))

    async def do_inspect_usage_demo(
        self,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        print("tool executed: inspect_usage_demo")
        return StepOutcome(
            {
                "status": "inspected",
                "next_action": "explain the observation in another turn",
            }
        )


class UsageConsoleSink:
    async def emit(self, event: AgentEvent) -> None:
        payload = event.payload
        if isinstance(payload, MessageCompleted):
            if payload.usage is None:
                print(f"turn {payload.turn} usage: unavailable")
            else:
                print(
                    f"turn {payload.turn} usage: "
                    f"input={payload.usage.input_tokens}, "
                    f"output={payload.usage.output_tokens}, "
                    f"total={payload.usage.total_tokens}"
                )
        elif isinstance(payload, AgentFinished):
            usage = payload.result.usage
            print(f"finished: {payload.result.status} / {payload.result.stop_reason}")
            print(
                f"run usage: total={usage.total_tokens}, "
                f"requests={usage.reported_request_count}/"
                f"{usage.request_count}, complete={usage.complete}"
            )


async def main() -> None:
    max_run_tokens = int(os.getenv("HARNESS_MAX_RUN_TOKENS", "1"))
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="usage-budget-demo",
        system_prompt=(
            "Call inspect_usage_demo exactly once. Wait for its result before "
            "producing any explanation. Do not call it more than once."
        ),
        handlers=[UsageTools()],
        runtime_policy=RuntimePolicy(
            max_steps=4,
            max_run_tokens=max_run_tokens,
            require_explicit_finish=True,
        ),
        event_sink=UsageConsoleSink(),
    )

    try:
        result = await agent.run(task="Inspect the usage budget demo.")
    finally:
        await agent.shutdown()

    print(f"configured max_run_tokens: {max_run_tokens}")
    if result.error:
        print(f"guard detail: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
