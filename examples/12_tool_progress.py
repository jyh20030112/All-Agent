"""Stream real tool execution progress through Harness events."""

import asyncio
from collections.abc import Mapping
from typing import Any

from ejagent import (
    AgentEvent,
    AgentFinished,
    BaseAgent,
    CancellationToken,
    MethodToolHandler,
    ModelConfig,
    OpenAIModelAdapter,
    RuntimePolicy,
    StepOutcome,
    ToolCompleted,
    ToolControl,
    ToolProgressed,
    ToolProgressReporter,
    ToolProgressUpdate,
    ToolStarted,
)

INDEX_TOOL = {
    "type": "function",
    "function": {
        "name": "index_project",
        "description": (
            "Index the demo project and complete the task. This tool must be "
            "called exactly once."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


class ProjectTools(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((INDEX_TOOL,))

    async def do_index_project(
        self,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        if progress is None:
            raise RuntimeError("tool progress reporter is unavailable")

        updates = (
            ToolProgressUpdate(
                "discovering files",
                {"completed": 1, "total": 3},
            ),
            ToolProgressUpdate(
                "parsing Python modules",
                {"completed": 2, "total": 3},
            ),
            ToolProgressUpdate(
                "building symbol index",
                {"completed": 3, "total": 3},
            ),
        )
        for update in updates:
            await progress.report(update)
            await asyncio.sleep(0.2)

        return StepOutcome(
            {"status": "indexed", "files": 24, "symbols": 137},
            control=ToolControl.COMPLETE,
        )


class ProgressConsoleSink:
    async def emit(self, event: AgentEvent) -> None:
        payload = event.payload
        if isinstance(payload, ToolStarted):
            print(f"tool started: {payload.tool_call.name}")
        elif isinstance(payload, ToolProgressed):
            print(f"  progress: {payload.update.message} {payload.update.data}")
        elif isinstance(payload, ToolCompleted):
            print(f"tool completed: {payload.tool_call.name}")
        elif isinstance(payload, AgentFinished):
            print(f"finished: {payload.result.status} / {payload.result.stop_reason}")


async def main() -> None:
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="tool-progress-demo",
        system_prompt=(
            "You demonstrate tool progress. Always call index_project exactly "
            "once for the user's request. Never answer with plain text."
        ),
        handlers=[ProjectTools()],
        runtime_policy=RuntimePolicy(
            max_steps=3,
            max_no_tool_responses=2,
            require_explicit_finish=True,
        ),
        event_sink=ProgressConsoleSink(),
    )

    try:
        result = await agent.run(task="Index the demo project now.")
        if not result.succeeded:
            result.raise_for_status()
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
