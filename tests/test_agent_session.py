import unittest
from typing import Any

from ejagent import (
    AgentEvent,
    AgentRunResult,
    AgentSession,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    CompositeAgentEventSink,
    MemorySessionStorage,
    MethodToolHandler,
    ModelAdapter,
    ModelToolCall,
    RunStatus,
    SessionRecorder,
    StepOutcome,
    StopReason,
)

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Return the provided text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


class SequenceModel(ModelAdapter):
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = list(responses)
        self.contexts: list[tuple[dict[str, Any], ...]] = []

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        self.contexts.append(context.agent_messages)
        return self.responses.pop(0)


class EchoHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((ECHO_TOOL,))

    async def do_echo(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome({"text": arguments.get("text")})


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class FailingSink:
    async def emit(self, event: AgentEvent) -> None:
        raise RuntimeError(f"failed for {event.kind}")


class AgentSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_recorder_persists_user_assistant_and_tool_messages(
        self,
    ) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="session-1", storage=storage)
        tool_call = ModelToolCall(
            id="call-1",
            name="echo",
            arguments='{"text": "hello"}',
        )
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(tool_calls=(tool_call,)),
                    AssistantMessage(content="finished"),
                ]
            ),
            agent_id="session-agent",
            handlers=[EchoHandler()],
            event_sink=recorder,
        )

        result = await agent.run(task="use echo")
        session = await recorder.load()

        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.agent_id, "session-agent")
        self.assertEqual(
            [message["role"] for message in session.messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertFalse(
            any(message["role"] == "system" for message in session.messages)
        )
        self.assertEqual(len(session.runs), 1)
        self.assertIs(session.runs[0].result, result)
        self.assertTrue(session.runs[0].finished)
        self.assertEqual(
            [entry.sequence for entry in session.entries],
            [1, 3, 5, 8],
        )

    async def test_one_session_records_multiple_runs(self) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="session-runs", storage=storage)
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(content="first answer"),
                    AssistantMessage(content="second answer"),
                ]
            ),
            agent_id="session-agent",
            event_sink=recorder,
        )

        first = await agent.run(task="first task")
        second = await agent.run(task="second task")
        session = await recorder.load()

        assert session is not None
        self.assertEqual(len(session.runs), 2)
        self.assertNotEqual(session.runs[0].run_id, session.runs[1].run_id)
        self.assertIs(session.runs[0].result, first)
        self.assertIs(session.runs[1].result, second)
        self.assertEqual(
            [message["content"] for message in session.messages],
            ["first task", "first answer", "second task", "second answer"],
        )

    async def test_loaded_history_can_resume_in_a_new_agent(self) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="resume", storage=storage)
        first_agent = BaseAgent(
            SequenceModel([AssistantMessage(content="saved answer")]),
            agent_id="resume-agent",
            event_sink=recorder,
        )
        await first_agent.run(task="saved task")
        session = await recorder.load()
        assert session is not None

        resumed_model = SequenceModel([AssistantMessage(content="continued answer")])
        resumed_agent = BaseAgent(
            resumed_model,
            agent_id="resume-agent",
        )
        resumed_agent.reset(session.messages)

        result = await resumed_agent.run(task="continue")

        self.assertEqual(result.output, "continued answer")
        self.assertEqual(
            [message.get("content") for message in resumed_model.contexts[0]],
            [
                resumed_agent.system_prompt,
                "saved task",
                "saved answer",
                "continue",
            ],
        )

    async def test_memory_storage_returns_detached_snapshots(self) -> None:
        storage = MemorySessionStorage()
        session = AgentSession(session_id="isolated")
        session.bind_agent("agent")
        session.begin_run("run-1", "task", 1)
        session.append_message(
            "run-1",
            2,
            {"role": "assistant", "content": "original"},
        )
        session.finish_run(
            "run-1",
            3,
            AgentRunResult(
                status=RunStatus.COMPLETED,
                stop_reason=StopReason.TEXT_RESPONSE,
                turns=1,
                output="original",
            ),
        )
        await storage.save(session)

        loaded = await storage.load("isolated")
        assert loaded is not None
        loaded.entries[1].message["content"] = "mutated"
        loaded.runs.clear()
        reloaded = await storage.load("isolated")

        assert reloaded is not None
        self.assertEqual(reloaded.messages[1]["content"], "original")
        self.assertEqual(len(reloaded.runs), 1)

    async def test_different_session_ids_are_isolated(self) -> None:
        storage = MemorySessionStorage()
        first_recorder = SessionRecorder(session_id="first", storage=storage)
        second_recorder = SessionRecorder(session_id="second", storage=storage)
        first_agent = BaseAgent(
            SequenceModel([AssistantMessage(content="one")]),
            agent_id="first-agent",
            event_sink=first_recorder,
        )
        second_agent = BaseAgent(
            SequenceModel([AssistantMessage(content="two")]),
            agent_id="second-agent",
            event_sink=second_recorder,
        )

        await first_agent.run(task="first task")
        await second_agent.run(task="second task")
        first_session = await first_recorder.load()
        second_session = await second_recorder.load()

        assert first_session is not None
        assert second_session is not None
        self.assertEqual(first_session.messages[0]["content"], "first task")
        self.assertEqual(second_session.messages[0]["content"], "second task")

    async def test_composite_sink_continues_after_one_observer_fails(
        self,
    ) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="composite", storage=storage)
        observer = RecordingSink()
        sink = CompositeAgentEventSink([FailingSink(), recorder, observer])
        agent = BaseAgent(
            SequenceModel([AssistantMessage(content="done")]),
            agent_id="composite-agent",
            event_sink=sink,
        )

        with self.assertLogs("composite-agent", level="WARNING"):
            result = await agent.run(task="record despite failure")
        session = await recorder.load()

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(len(observer.events), 5)
        assert session is not None
        self.assertEqual(
            [message["content"] for message in session.messages],
            ["record despite failure", "done"],
        )


if __name__ == "__main__":
    unittest.main()
