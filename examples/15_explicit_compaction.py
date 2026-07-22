"""Explicitly compact history with a real Provider-backed Compactor."""

import asyncio
import json
import os

from ejagent import (
    AgentEvent,
    BaseAgent,
    CompactionCompleted,
    CompactionFailed,
    CompactionPolicy,
    CompactionRequest,
    CompactionStarted,
    CompositeAgentEventSink,
    ContextBudget,
    ContextBuildResult,
    MemorySessionStorage,
    ModelCompactor,
    ModelConfig,
    OpenAIModelAdapter,
    SessionRecorder,
)


def build_compaction_context(request: CompactionRequest) -> ContextBuildResult:
    """Application-owned prompt policy injected into ModelCompactor."""

    visible_messages = [
        {
            key: value
            for key, value in message.items()
            if key not in {"usage", "_ejagent_summary", "_simagentplg_summary"}
        }
        for message in request.preparation.messages_to_summarize
    ]
    previous = (
        request.previous_summary.content
        if request.previous_summary is not None
        else "(none)"
    )
    prompt = (
        "Create a concise continuation summary. Preserve user goals, "
        "tool findings, decisions, and unfinished work. Do not invent "
        "facts. Return only the summary.\n\n"
        f"Previous summary:\n{previous}\n\n"
        "New messages:\n" + json.dumps(visible_messages, ensure_ascii=False, indent=2)
    )
    messages = (
        {
            "role": "system",
            "content": "You produce reliable Agent context summaries.",
        },
        {"role": "user", "content": prompt},
    )
    return ContextBuildResult(
        agent_messages=messages,
        llm_messages=messages,
        tools=(),
    )


class CompactionConsoleSink:
    async def emit(self, event: AgentEvent) -> None:
        payload = event.payload
        if isinstance(payload, CompactionStarted):
            preparation = payload.request.preparation
            print(
                "compaction started: "
                f"summarize={len(preparation.messages_to_summarize)}, "
                f"keep={len(preparation.messages_to_keep)}"
            )
        elif isinstance(payload, CompactionCompleted):
            print(f"compaction completed: {payload.result.status}")
        elif isinstance(payload, CompactionFailed):
            print(
                f"compaction failed: {payload.result.status}, "
                f"error={payload.result.error}"
            )


OLD_TOOL_OUTPUT = "\n".join(
    f"diagnostic line {index}: repeated legacy warning" for index in range(80)
)

HISTORY = [
    {"role": "user", "content": "Inspect the legacy report."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "legacy-call",
                "type": "function",
                "function": {
                    "name": "read_report",
                    "arguments": '{"path":"legacy.txt"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "legacy-call",
        "content": OLD_TOOL_OUTPUT,
    },
    {
        "role": "assistant",
        "content": "The legacy report contains repeated warnings.",
    },
    {
        "role": "user",
        "content": "Keep that finding available for the next question.",
    },
    {
        "role": "assistant",
        "content": "The finding will remain available.",
    },
]


async def main() -> None:
    model = OpenAIModelAdapter(ModelConfig.from_env())
    storage = MemorySessionStorage()
    recorder = SessionRecorder(
        session_id="explicit-compaction-demo",
        storage=storage,
    )
    agent = BaseAgent(
        model,
        agent_id="explicit-compaction-demo",
        system_prompt=(
            "Answer using the retained conversation summary and recent history."
        ),
        compaction_policy=CompactionPolicy(
            ContextBudget(
                context_window=int(os.getenv("HARNESS_CONTEXT_WINDOW", "4096")),
                reserve_tokens=int(os.getenv("HARNESS_CONTEXT_RESERVE", "512")),
                keep_recent_tokens=int(os.getenv("HARNESS_KEEP_RECENT_TOKENS", "20")),
            )
        ),
        compactor=ModelCompactor(
            model,
            context_builder=build_compaction_context,
            source=f"openai-compatible:{model.config.model}",
        ),
        event_sink=CompositeAgentEventSink([recorder, CompactionConsoleSink()]),
    )
    agent.reset(history=HISTORY)

    try:
        compacted = await agent.compact()
        print(f"summary source: {compacted.summary.source}")
        print(f"state roles after compact: {[m['role'] for m in agent.messages]}")
        result = await agent.run(
            task="What did the legacy report contain? Reply in one sentence."
        )
        session = await recorder.load()
    finally:
        await agent.shutdown()

    print(f"agent answer: {result.output}")
    assert session is not None
    print(f"session compactions: {len(session.compactions)}")
    print(f"session roles: {[message['role'] for message in session.messages]}")


if __name__ == "__main__":
    asyncio.run(main())
