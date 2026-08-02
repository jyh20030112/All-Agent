"""Compose an AgentHarness with a custom atomic math tool."""

import asyncio

from ejagent.contracts import (
    CancellationToken,
    ToolCall,
    ToolControl,
    ToolDefinition,
    ToolExecutionResult,
    ToolSemantics,
)
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionTool, FunctionToolExecutor

ADD_TOOL = ToolDefinition(
    name="add",
    description="Add two numbers.",
    input_schema={
        "type": "object",
        "properties": {
            "left": {"type": "number"},
            "right": {"type": "number"},
        },
        "required": ["left", "right"],
    },
    semantics=ToolSemantics.read_only(),
)


async def add(
    call: ToolCall,
    cancellation: CancellationToken,
) -> ToolExecutionResult:
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
    value = left + right
    return ToolExecutionResult(
        {"status": "success", "value": value},
        control=ToolControl.COMPLETE,
        output=str(value),
    )


async def main() -> None:
    harness = AgentHarness(
        agent_id="calculator",
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=FunctionToolExecutor((FunctionTool(ADD_TOOL, add),)),
    )

    async with harness:
        outcome = await harness.run("Use the add tool to calculate 19.5 + 22.5.")
        print(outcome.result.output)


if __name__ == "__main__":
    asyncio.run(main())
