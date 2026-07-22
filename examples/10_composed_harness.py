"""Compose the real provider, tools, events, Session, and runtime policy."""

import asyncio
from collections.abc import Mapping
from typing import Any

from ejagent import (
    AgentEvent,
    BaseAgent,
    CancellationToken,
    CompositeAgentEventSink,
    MemorySessionStorage,
    MethodToolHandler,
    ModelConfig,
    OpenAIModelAdapter,
    RuntimePolicy,
    SessionRecorder,
    StepOutcome,
    ToolCallContext,
    ToolControl,
    ToolMiddleware,
    ToolNext,
)

INSPECT_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_project",
        "description": "Return current EJAgent Core Harness metadata.",
        "parameters": {"type": "object", "properties": {}},
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Explicitly finish after inspecting the project.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
}


class HarnessTools(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((INSPECT_TOOL, FINISH_TOOL))
        self.inspected = False

    async def on_task_start(self) -> None:
        self.inspected = False

    async def do_inspect_project(
        self,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        self.inspected = True
        return StepOutcome(
            {
                "agent_core": "EJAgent Core",
                "orchestrator": "AgentOrchestrator",
                "events": "AgentEvent + CompositeAgentEventSink",
                "session": "SessionRecorder + MemorySessionStorage",
                "runtime_control": "CancellationToken + abort + wait_for_idle",
            }
        )

    async def do_finish(
        self,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        if not self.inspected:
            return StepOutcome(
                {
                    "status": "error",
                    "error": "inspect_project must be called before finish",
                }
            )
        return StepOutcome(
            {"summary": arguments["summary"]},
            control=ToolControl.COMPLETE,
        )


class ToolAuditMiddleware(ToolMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def __call__(
        self,
        context: ToolCallContext,
        call_next: ToolNext,
    ) -> StepOutcome:
        if context.cancellation is None:
            raise RuntimeError("tool call has no cancellation token")
        self.calls.append(context.tool_name)
        print(f"middleware before: {context.tool_name}")
        outcome = await call_next(context)
        print(f"middleware after: {context.tool_name}")
        return outcome


class ConsoleEventSink:
    async def emit(self, event: AgentEvent) -> None:
        print(f"event #{event.sequence}: {event.kind}")


async def main() -> None:
    storage = MemorySessionStorage()
    recorder = SessionRecorder(
        session_id="composed-harness",
        storage=storage,
    )
    middleware = ToolAuditMiddleware()
    event_sink = CompositeAgentEventSink([ConsoleEventSink(), recorder])
    agent = BaseAgent(
        OpenAIModelAdapter(ModelConfig.from_env()),
        agent_id="composed-harness",
        system_prompt=(
            "You are demonstrating an Agent Harness. First call "
            "inspect_project exactly once. Then call finish with a concise "
            "summary of the returned metadata. Never finish with plain text."
        ),
        handlers=[HarnessTools()],
        middlewares=[middleware],
        runtime_policy=RuntimePolicy(
            max_steps=6,
            max_no_tool_responses=2,
            require_explicit_finish=True,
        ),
        event_sink=event_sink,
    )

    try:
        result = await agent.run(
            task="Inspect the registered Harness metadata and finish."
        )
    finally:
        await agent.shutdown()

    session = await recorder.load()
    if session is None:
        raise RuntimeError("session was not recorded")

    print(f"result: {result.status} / {result.stop_reason}")
    print(f"output: {result.output}")
    print(f"tool calls: {middleware.calls}")
    print(f"session roles: {[message['role'] for message in session.messages]}")
    print(f"session runs: {len(session.runs)}")


if __name__ == "__main__":
    asyncio.run(main())
