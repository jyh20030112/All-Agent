import asyncio
import inspect
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from simagentplg import (
    AgentContextBuilder,
    AgentOrchestrator,
    AgentRunError,
    AgentRunResult,
    AgentState,
    AgentStatus,
    BaseAgent,
    CancellationToken,
    McpToolHandler,
    MethodToolHandler,
    Middleware,
    ModelAdapter,
    ModelConfig,
    OpenAIModelAdapter,
    RunStatus,
    RuntimePolicy,
    StepOutcome,
    StopReason,
    ToolCallContext,
    ToolControl,
    ToolMiddleware,
    ToolNext,
)
from simagentplg.agent.base import (
    DEFAULT_SYSTEM_PROMPT,
    EXPLICIT_FINISH_PROTOCOL_PROMPT,
    TOOL_PROTOCOL_PROMPT,
)
from simagentplg.agent.orchestrator import TOOL_COMPLETION_RETRY_PROMPT

TEST_CONFIG = ModelConfig(
    model="test-model",
    api_key="test-key",
    base_url="https://example.invalid",
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

DONE_TOOL = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "Finish the current test task.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
}


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"

    @property
    def name(self) -> str:
        return self.function.name

    @property
    def arguments(self) -> str:
        return self.function.arguments

    def to_agent_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


class FakeMessage:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[FakeToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def to_agent_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = [
                tool_call.to_agent_message() for tool_call in self.tool_calls
            ]
        return message


class FakeCompletions:
    def __init__(self, responses: list[FakeMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeMessage:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeModelAdapter(ModelAdapter):
    def __init__(self, responses: list[FakeMessage]) -> None:
        self.completions = FakeCompletions(responses)
        self.started = 0
        self.stopped = 0

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> FakeMessage:
        return await self.completions.create(
            messages=context.llm_messages,
            tools=context.tools or None,
        )


class EchoHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((ECHO_TOOL,))
        self.started = 0
        self.stopped = 0
        self.task_starts = 0
        self.calls = 0

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1

    async def on_task_start(self) -> None:
        self.task_starts += 1

    async def do_echo(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        self.calls += 1
        text = arguments.get("text")
        if not isinstance(text, str):
            return StepOutcome({"status": "error", "error": "text is required"})
        return StepOutcome({"status": "success", "text": text})


class DoneHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((DONE_TOOL,))

    async def do_done(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome(
            {"summary": arguments.get("summary", "")},
            control=ToolControl.COMPLETE,
        )


class FakeMcpManager:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def startup(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.stopped = True

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "demo__lookup",
                    "description": "Lookup a value.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        self.calls.append((tool_name, arguments))
        return "mcp-result"


class RecordingToolMiddleware(ToolMiddleware):
    def __init__(
        self,
        outcome: StepOutcome | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.outcome = outcome
        self.started = 0
        self.stopped = 0
        self.task_starts = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.contexts: list[ToolCallContext] = []

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1

    async def on_task_start(self) -> None:
        self.task_starts += 1

    async def __call__(
        self,
        context: ToolCallContext,
        call_next: ToolNext,
    ) -> StepOutcome:
        self.calls.append((context.tool_name, dict(context.arguments)))
        self.contexts.append(context)
        if self.outcome is not None:
            return self.outcome
        return await call_next(context)


def make_agent(
    *,
    model: ModelAdapter | None = None,
    **kwargs: Any,
) -> BaseAgent:
    """Build a core agent while keeping repetitive test setup compact."""

    return BaseAgent(
        model or FakeModelAdapter([]),
        **kwargs,
    )


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_base_agent_composes_public_orchestrator(self) -> None:
        agent = make_agent(
            agent_id="orchestrated",
            model=FakeModelAdapter([]),
        )

        self.assertIsInstance(agent.orchestrator, AgentOrchestrator)
        self.assertIs(agent.orchestrator.state, agent.state)

    async def test_middleware_base_class_is_exported_with_standard_spelling(
        self,
    ) -> None:
        self.assertTrue(issubclass(ToolMiddleware, Middleware))

    async def test_base_agent_public_api_uses_model_and_runtime_policy(self) -> None:
        parameters = inspect.signature(BaseAgent).parameters

        self.assertIn("model", parameters)
        self.assertIn("runtime_policy", parameters)
        self.assertNotIn("client", parameters)
        self.assertNotIn("max_steps", parameters)
        self.assertNotIn(
            "has_handler_tools",
            inspect.signature(AgentOrchestrator).parameters,
        )

    async def test_base_agent_owns_model_adapter_lifecycle(self) -> None:
        model = FakeModelAdapter([])
        agent = BaseAgent(model, agent_id="model-lifecycle")

        await agent.startup()
        await agent.startup()
        await agent.shutdown()
        await agent.shutdown()

        self.assertEqual(model.started, 1)
        self.assertEqual(model.stopped, 1)

    async def test_model_config_reads_chat_model_from_env(self) -> None:
        with (
            patch("simagentplg.providers.openai.load_dotenv"),
            patch.dict(
                "os.environ",
                {
                    "CHAT_MODEL": "chat-model",
                    "MODEL_API_KEY": "key",
                    "MODEL_URL": "https://model.example",
                    "LLM_TIMEOUT": "12",
                    "LLM_TEMPERATURE": "0.2",
                },
                clear=True,
            ),
        ):
            config = ModelConfig.from_env()

        self.assertEqual(config.model, "chat-model")
        self.assertEqual(config.api_key, "key")
        self.assertEqual(config.base_url, "https://model.example")
        self.assertEqual(config.timeout, 12)
        self.assertEqual(config.temperature, 0.2)

    async def test_openai_adapter_converts_provider_response(self) -> None:
        class OpenAICompletions:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            async def create(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=FakeMessage(
                                tool_calls=[
                                    FakeToolCall(
                                        id="call-1",
                                        function=FakeFunction(
                                            "lookup",
                                            '{"query": "value"}',
                                        ),
                                    )
                                ]
                            )
                        )
                    ]
                )

        completions = OpenAICompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = OpenAIModelAdapter(
            TEST_CONFIG,
            client=client,  # type: ignore[arg-type]
        )
        context = AgentContextBuilder().build(AgentState())

        message = await adapter.complete(context)

        self.assertIsNone(message.content)
        self.assertEqual(message.tool_calls[0].id, "call-1")
        self.assertEqual(message.tool_calls[0].name, "lookup")
        self.assertEqual(
            message.tool_calls[0].arguments,
            '{"query": "value"}',
        )
        self.assertEqual(completions.calls[0]["model"], "test-model")

    async def test_agents_use_independent_models_and_messages(self) -> None:
        first_model = FakeModelAdapter([FakeMessage("first")])
        second_model = FakeModelAdapter([FakeMessage("second")])
        first = make_agent(
            agent_id="first",
            model=first_model,
        )
        second = make_agent(
            agent_id="second",
            model=second_model,
        )

        await first.runtime(task="only first sees this")
        await second.runtime(task="only second sees this")

        self.assertIsNot(first.model, second.model)
        self.assertNotEqual(first.messages, second.messages)
        self.assertFalse(
            any(
                message.get("content") == "only first sees this"
                for message in second.messages
            )
        )

    async def test_runtime_keeps_memory_and_reset_clears_it(self) -> None:
        client = FakeModelAdapter([FakeMessage("one"), FakeMessage("two")])
        agent = make_agent(
            agent_id="memory",
            model=client,
        )

        await agent.runtime(task="first task")
        await agent.runtime(task="second task")

        second_call_messages = client.completions.calls[1]["messages"]
        self.assertTrue(
            any(message.get("content") == "one" for message in second_call_messages)
        )

        agent.reset([{"role": "user", "content": "seed"}])
        self.assertEqual(
            agent.messages,
            [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": "seed"},
            ],
        )

    async def test_runtime_calls_are_serialized_per_agent(self) -> None:
        agent = make_agent(
            agent_id="serialized-runtime",
            model=FakeModelAdapter([]),
        )
        active = 0
        maximum = 0

        async def run_loop(
            cancellation: CancellationToken,
            steering: object | None = None,
            *,
            run_id: str,
        ) -> AgentRunResult:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.01)
                return AgentRunResult(
                    status=RunStatus.COMPLETED,
                    stop_reason=StopReason.TEXT_RESPONSE,
                    turns=1,
                    output="done",
                )
            finally:
                active -= 1

        agent.orchestrator._run_loop = run_loop  # type: ignore[method-assign]

        results = await asyncio.gather(
            agent.runtime(task="first"),
            agent.runtime(task="second"),
        )

        self.assertEqual(results, ["done", "done"])
        self.assertEqual(maximum, 1)

    async def test_default_system_prompt_is_plain_chat_prompt(self) -> None:
        agent = make_agent(
            agent_id="default-prompt",
            model=FakeModelAdapter([]),
        )

        self.assertEqual(agent.system_prompt, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(
            agent.messages,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
        )

    async def test_tool_mode_injects_tool_protocol_explicitly(self) -> None:
        agent = make_agent(
            agent_id="tool-protocol",
            system_prompt="You are a custom coding agent.",
            handlers=[DoneHandler()],
            model=FakeModelAdapter([]),
        )

        self.assertEqual(
            agent.messages,
            [
                {"role": "system", "content": "You are a custom coding agent."},
                {"role": "system", "content": TOOL_PROTOCOL_PROMPT},
            ],
        )

    async def test_plain_chat_has_no_tool_protocol_message(self) -> None:
        agent = make_agent(
            agent_id="plain-protocol",
            system_prompt="You are a plain chat agent.",
            model=FakeModelAdapter([]),
        )

        self.assertEqual(
            agent.messages,
            [
                {"role": "system", "content": "You are a plain chat agent."},
            ],
        )

    async def test_explicit_finish_policy_injects_completion_protocol(self) -> None:
        agent = make_agent(
            agent_id="explicit-finish-protocol",
            handlers=[DoneHandler()],
            runtime_policy=RuntimePolicy(require_explicit_finish=True),
            model=FakeModelAdapter([]),
        )

        self.assertEqual(
            agent.messages[-1],
            {"role": "system", "content": EXPLICIT_FINISH_PROTOCOL_PROMPT},
        )

    async def test_tools_do_not_require_explicit_finish_by_default(self) -> None:
        agent = make_agent(
            agent_id="tool-text-completion",
            handlers=[EchoHandler()],
            model=FakeModelAdapter([FakeMessage("completed with plain text")]),
        )

        result = await agent.runtime(task="answer after using tools if needed")

        self.assertEqual(result, "completed with plain text")

    async def test_run_returns_structured_result(self) -> None:
        agent = make_agent(
            agent_id="structured-run",
            model=FakeModelAdapter([FakeMessage("done")]),
        )

        result = await agent.run(task="complete")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.stop_reason, StopReason.TEXT_RESPONSE)
        self.assertEqual(result.output, "done")
        self.assertEqual(result.turns, 1)

    async def test_plain_chat_returns_text_without_retry_prompt(self) -> None:
        client = FakeModelAdapter([FakeMessage("plain text")])
        agent = make_agent(
            agent_id="plain-retry",
            runtime_policy=RuntimePolicy(max_steps=2),
            model=client,
        )

        result = await agent.runtime(task="plain chat")

        self.assertEqual(result, "plain text")
        self.assertFalse(
            any(
                message.get("content") == TOOL_COMPLETION_RETRY_PROMPT
                for message in agent.messages
            )
        )

    async def test_plain_chat_rejects_empty_completion(self) -> None:
        client = FakeModelAdapter([FakeMessage(None)])
        agent = make_agent(
            agent_id="plain-empty",
            model=client,
        )

        with self.assertRaisesRegex(RuntimeError, "empty content"):
            await agent.runtime(task="plain chat")

    async def test_plain_chat_raises_when_step_limit_is_exhausted(self) -> None:
        unknown_call = FakeToolCall(
            id="call-1",
            function=FakeFunction("unknown", "{}"),
        )
        client = FakeModelAdapter(
            [
                FakeMessage(tool_calls=[unknown_call]),
                FakeMessage(tool_calls=[unknown_call]),
            ]
        )
        agent = make_agent(
            agent_id="plain-step-limit",
            runtime_policy=RuntimePolicy(max_steps=2),
            model=client,
        )

        with self.assertRaisesRegex(RuntimeError, "did not finish within 2"):
            await agent.runtime(task="plain chat")

    async def test_context_builder_can_filter_internal_context(self) -> None:
        class FilteringContextBuilder(AgentContextBuilder):
            def convert_to_llm_messages(
                self,
                messages: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                return [
                    dict(message)
                    for message in messages
                    if not message.get("exclude_from_llm")
                ]

        client = FakeModelAdapter([FakeMessage("visible")])
        agent = make_agent(
            agent_id="context",
            model=client,
            context_builder=FilteringContextBuilder(),
        )
        agent.messages.append(
            {
                "role": "user",
                "content": "internal note",
                "exclude_from_llm": True,
            }
        )

        result = await agent.runtime(task="real task")

        self.assertEqual(result, "visible")
        sent_messages = client.completions.calls[0]["messages"]
        self.assertFalse(
            any(message.get("content") == "internal note" for message in sent_messages)
        )
        self.assertTrue(
            any(message.get("content") == "real task" for message in sent_messages)
        )

    async def test_agent_state_records_completed_task(self) -> None:
        agent = make_agent(
            agent_id="state-complete",
            model=FakeModelAdapter([FakeMessage("done")]),
        )

        result = await agent.runtime(task="finish task")

        self.assertEqual(result, "done")
        self.assertEqual(agent.state.task, "finish task")
        self.assertEqual(agent.state.status, AgentStatus.COMPLETED)
        self.assertEqual(agent.state.turn, 1)
        self.assertEqual(agent.state.result, "done")
        self.assertIsNone(agent.state.error)

    async def test_agent_state_records_failed_task(self) -> None:
        agent = make_agent(
            agent_id="state-failed",
            model=FakeModelAdapter([FakeMessage(None)]),
        )

        with self.assertRaisesRegex(RuntimeError, "empty content"):
            await agent.runtime(task="fail task")

        self.assertEqual(agent.state.task, "fail task")
        self.assertEqual(agent.state.status, AgentStatus.FAILED)
        self.assertIsNone(agent.state.result)
        self.assertIn("empty content", agent.state.error or "")

    async def test_context_builder_does_not_mutate_agent_state(self) -> None:
        state = AgentState(messages=[{"role": "user", "content": "saved"}])

        result = AgentContextBuilder().build(
            state,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "temporary_tool"},
                }
            ],
            transient_messages=[{"role": "system", "content": "temporary"}],
        )

        self.assertEqual(
            state.messages,
            [{"role": "user", "content": "saved"}],
        )
        self.assertEqual(
            result.llm_messages,
            (
                {"role": "user", "content": "saved"},
                {"role": "system", "content": "temporary"},
            ),
        )
        self.assertEqual(
            result.tools,
            (
                {
                    "type": "function",
                    "function": {"name": "temporary_tool"},
                },
            ),
        )

    async def test_skill_metadata_is_injected_without_llm_router(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir)
            skill_dir = skills_dir / "release_notes"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: release_notes",
                        "description: Write user-facing release notes.",
                        "---",
                        "",
                        "# Release Notes",
                        "",
                        "This full instruction should load only on demand.",
                    ]
                ),
                encoding="utf-8",
            )
            client = FakeModelAdapter([FakeMessage("ok")])
            agent = make_agent(
                agent_id="skills-index",
                skills_dir=skills_dir,
                model=client,
            )

            with patch.dict("os.environ", {}, clear=True):
                result = await agent.runtime(task="Write release notes")

        self.assertEqual(result, "ok")
        sent_messages = client.completions.calls[0]["messages"]
        joined = "\n".join(str(message.get("content", "")) for message in sent_messages)
        self.assertIn("Available skills:", joined)
        self.assertIn("name: release_notes", joined)
        self.assertIn("description: Write user-facing release notes.", joined)
        self.assertIn(
            f"location: {(skill_dir / 'SKILL.md').resolve()}",
            joined,
        )
        self.assertNotIn("This full instruction should load only on demand.", joined)

    async def test_skill_discovery_is_not_repeated_between_runtime_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir)
            skill_dir = skills_dir / "release_notes"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: release_notes",
                        "description: Write release notes.",
                        "---",
                    ]
                ),
                encoding="utf-8",
            )
            client = FakeModelAdapter([FakeMessage("first"), FakeMessage("second")])
            agent = make_agent(
                agent_id="skills-once",
                skills_dir=skills_dir,
                model=client,
            )

            await agent.runtime(task="first")
            new_skill_dir = skills_dir / "hot_loaded"
            new_skill_dir.mkdir()
            (new_skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: hot_loaded",
                        "description: Should not appear without refresh.",
                        "---",
                    ]
                ),
                encoding="utf-8",
            )
            await agent.runtime(task="second")

        self.assertIsNone(client.completions.calls[1]["tools"])
        second_messages = client.completions.calls[1]["messages"]
        joined = "\n".join(
            str(message.get("content", "")) for message in second_messages
        )
        self.assertIn("name: release_notes", joined)
        self.assertNotIn("name: hot_loaded", joined)

    async def test_explicit_skill_name_loads_full_skill_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir)
            skill_dir = skills_dir / "release_notes"
            examples_dir = skill_dir / "examples"
            examples_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: release_notes",
                        "description: Write user-facing release notes.",
                        "---",
                        "",
                        "# Release Notes",
                        "",
                        "FULL SKILL RULES",
                    ]
                ),
                encoding="utf-8",
            )
            (skill_dir / "template.md").write_text("TEMPLATE BODY", encoding="utf-8")
            (examples_dir / "sample.md").write_text("SAMPLE BODY", encoding="utf-8")
            client = FakeModelAdapter([FakeMessage("ok")])
            agent = make_agent(
                agent_id="skills-load",
                skills_dir=skills_dir,
                model=client,
            )

            await agent.runtime(task="$release_notes Write release notes")

        sent_messages = client.completions.calls[0]["messages"]
        joined = "\n".join(str(message.get("content", "")) for message in sent_messages)
        self.assertIn('You are executing the local skill "release_notes".', joined)
        self.assertIn("FULL SKILL RULES", joined)
        self.assertIn("TEMPLATE BODY", joined)
        self.assertIn("SAMPLE BODY", joined)

    async def test_skills_do_not_register_an_internal_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir)
            skill_dir = skills_dir / "release_notes"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: release_notes",
                        "description: Write user-facing release notes.",
                        "---",
                        "",
                        "# Release Notes",
                        "",
                        "FULL SKILL RULES",
                    ]
                ),
                encoding="utf-8",
            )
            client = FakeModelAdapter([FakeMessage("release notes")])
            agent = make_agent(
                agent_id="skills-resource",
                skills_dir=skills_dir,
                model=client,
            )

            result = await agent.runtime(task="Write release notes")

        self.assertEqual(result, "release notes")
        self.assertEqual(agent.tools, [])
        self.assertIsNone(client.completions.calls[0]["tools"])
        self.assertFalse(
            any(message.get("role") == "tool" for message in agent.messages)
        )

    async def test_full_skill_context_file_reads_are_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir)
            skill_dir = skills_dir / "release_notes"
            examples_dir = skill_dir / "examples"
            examples_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: release_notes",
                        "description: Write user-facing release notes.",
                        "---",
                        "",
                        "# Release Notes",
                        "",
                        "FULL SKILL RULES",
                    ]
                ),
                encoding="utf-8",
            )
            (skill_dir / "template.md").write_text("TEMPLATE BODY", encoding="utf-8")
            (examples_dir / "sample.md").write_text("SAMPLE BODY", encoding="utf-8")
            client = FakeModelAdapter([FakeMessage("release notes")])
            agent = make_agent(
                agent_id="skills-cache",
                skills_dir=skills_dir,
                model=client,
            )
            await agent.startup()

            original_read_text = Path.read_text
            read_counts: dict[Path, int] = {}

            def count_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
                read_counts[path] = read_counts.get(path, 0) + 1
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, "read_text", autospec=True) as read_text:
                read_text.side_effect = count_read_text
                result = await agent.runtime(task="$release_notes Write release notes")

        self.assertEqual(result, "release notes")
        self.assertEqual(read_counts[skill_dir / "SKILL.md"], 1)
        self.assertEqual(read_counts[skill_dir / "template.md"], 1)
        self.assertEqual(read_counts[examples_dir / "sample.md"], 1)

    async def test_agent_id_is_normalized_and_read_only(self) -> None:
        agent = make_agent(
            agent_id="  assistant  ",
            model=FakeModelAdapter([]),
        )

        self.assertEqual(agent.agent_id, "assistant")
        with self.assertRaises(AttributeError):
            agent.agent_id = "renamed"  # type: ignore[misc]

    async def test_empty_agent_id_is_rejected(self) -> None:
        for agent_id in ("", "   "):
            with self.subTest(agent_id=agent_id):
                with self.assertRaisesRegex(ValueError, "agent_id"):
                    make_agent(
                        agent_id=agent_id,
                        model=FakeModelAdapter([]),
                    )

    async def test_agent_id_is_required(self) -> None:
        with self.assertRaisesRegex(TypeError, "agent_id"):
            BaseAgent(FakeModelAdapter([]))  # type: ignore[call-arg]

    async def test_method_handler_dispatches_atomic_tool(self) -> None:
        handler = EchoHandler()
        result = await handler.dispatch("echo", {"text": "hello"})
        self.assertEqual(
            result.data,
            {"status": "success", "text": "hello"},
        )

    async def test_tool_mode_uses_only_explicit_handlers(self) -> None:
        echo = EchoHandler()
        agent = make_agent(
            agent_id="tools",
            handlers=[echo],
            model=FakeModelAdapter([]),
        )

        self.assertEqual(agent.handlers, [echo])
        self.assertEqual(
            [tool["function"]["name"] for tool in agent.tools],
            ["echo"],
        )

    async def test_tool_disabled_mode_does_not_add_bash_handler(self) -> None:
        agent = make_agent(
            agent_id="chat",
            model=FakeModelAdapter([]),
        )

        self.assertEqual(agent.handlers, [])
        self.assertEqual(agent.tools, [])

    async def test_duplicate_tool_names_fail_during_startup(self) -> None:
        model = FakeModelAdapter([])
        agent = make_agent(
            agent_id="duplicate-tools",
            handlers=[EchoHandler(), EchoHandler()],
            model=model,
        )

        with self.assertRaisesRegex(ValueError, "duplicate tool 'echo'"):
            await agent.startup()

        self.assertEqual(model.started, 1)
        self.assertEqual(model.stopped, 1)

    async def test_unknown_tool_has_explicit_error(self) -> None:
        agent = make_agent(
            agent_id="unknown-tool",
            handlers=[EchoHandler()],
            model=FakeModelAdapter([]),
        )

        with self.assertRaisesRegex(KeyError, "unknown tool 'missing'"):
            await agent.dispatch("missing", {})

    async def test_invalid_json_tool_arguments_are_returned_to_model(self) -> None:
        client = FakeModelAdapter(
            [
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-1",
                            function=FakeFunction("echo", "[1, 2]"),
                        )
                    ]
                ),
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-2",
                            function=FakeFunction(
                                "done",
                                '{"summary": "recovered"}',
                            ),
                        )
                    ]
                ),
            ]
        )
        agent = make_agent(
            agent_id="invalid-json",
            handlers=[EchoHandler(), DoneHandler()],
            model=client,
        )

        result = await agent.runtime(task="use echo")

        self.assertEqual(json.loads(result or "")["summary"], "recovered")
        tool_message = next(
            message for message in agent.messages if message["role"] == "tool"
        )
        payload = json.loads(tool_message["content"])
        self.assertEqual(payload["status"], "error")
        self.assertIn("JSON object", payload["error"])

    async def test_tool_mode_corrects_plain_text_until_exit_tool(self) -> None:
        client = FakeModelAdapter(
            [
                FakeMessage("premature answer"),
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-1",
                            function=FakeFunction(
                                "done",
                                '{"summary": "done"}',
                            ),
                        )
                    ]
                ),
            ]
        )
        agent = make_agent(
            agent_id="finish-required",
            handlers=[DoneHandler()],
            runtime_policy=RuntimePolicy(require_explicit_finish=True),
            model=client,
        )

        result = await agent.runtime(task="complete the task")

        self.assertEqual(json.loads(result or "")["summary"], "done")
        second_messages = client.completions.calls[1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and message.get("content") == TOOL_COMPLETION_RETRY_PROMPT
                for message in second_messages
            )
        )

    async def test_tool_mode_retries_plain_text_with_diagnostic_limit(self) -> None:
        client = FakeModelAdapter(
            [
                FakeMessage("first plain text"),
                FakeMessage("second plain text"),
                FakeMessage("third plain text"),
            ]
        )
        agent = make_agent(
            agent_id="tool-retry-limit",
            handlers=[DoneHandler()],
            runtime_policy=RuntimePolicy(
                max_steps=5,
                require_explicit_finish=True,
            ),
            model=client,
        )

        with self.assertRaisesRegex(RuntimeError, "without a completing tool call"):
            await agent.runtime(task="complete the task")

        retry_prompts = [
            message["content"]
            for call in client.completions.calls[1:]
            for message in call["messages"]
            if message.get("role") == "system"
            and message.get("content", "").startswith(
                "Explicit-finish mode requires a completing tool call"
            )
        ]
        self.assertEqual(len(retry_prompts), 2)
        self.assertEqual(len(set(retry_prompts)), 2)
        self.assertIn("Retry 2/3", retry_prompts[1])
        self.assertFalse(
            any(
                message.get("content", "").startswith(
                    "Explicit-finish mode requires a completing tool call"
                )
                for message in agent.state.messages
            )
        )

    async def test_tool_lifecycle_is_not_duplicated_in_logs(self) -> None:
        client = FakeModelAdapter(
            [
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-1",
                            function=FakeFunction(
                                "echo",
                                '{"text": "hello"}',
                            ),
                        )
                    ]
                ),
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-2",
                            function=FakeFunction(
                                "done",
                                '{"summary": "logged"}',
                            ),
                        )
                    ]
                ),
            ]
        )
        agent = make_agent(
            agent_id="log-tools",
            handlers=[EchoHandler(), DoneHandler()],
            model=client,
        )

        with self.assertLogs("log-tools", level="INFO") as captured:
            result = await agent.runtime(task="use echo then finish")

        self.assertEqual(json.loads(result or "")["summary"], "logged")
        logs = "\n".join(captured.output)
        self.assertIn("registered tools: done, echo", logs)
        self.assertNotIn("Calling tool", logs)
        self.assertNotIn("Tool echo completed", logs)
        self.assertNotIn("Tool done completed", logs)

    async def test_task_start_hook_runs_for_each_tool_task(self) -> None:
        handler = EchoHandler()
        client = FakeModelAdapter(
            [
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-1",
                            function=FakeFunction(
                                "done",
                                '{"summary": "first"}',
                            ),
                        )
                    ]
                ),
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-2",
                            function=FakeFunction(
                                "done",
                                '{"summary": "second"}',
                            ),
                        )
                    ]
                ),
            ]
        )
        agent = make_agent(
            agent_id="task-hooks",
            handlers=[handler, DoneHandler()],
            model=client,
        )

        await agent.runtime(task="first")
        await agent.runtime(task="second")

        self.assertEqual(handler.task_starts, 2)

    async def test_startup_before_runtime_does_not_repeat_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir)
            skill_dir = skills_dir / "release_notes"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: release_notes",
                        "description: Write release notes.",
                        "---",
                    ]
                ),
                encoding="utf-8",
            )
            handler = EchoHandler()
            client = FakeModelAdapter(
                [
                    FakeMessage(
                        tool_calls=[
                            FakeToolCall(
                                id="call-1",
                                function=FakeFunction(
                                    "done",
                                    '{"summary": "done"}',
                                ),
                            )
                        ]
                    )
                ]
            )
            agent = make_agent(
                agent_id="startup-once",
                handlers=[handler, DoneHandler()],
                skills_dir=skills_dir,
                model=client,
            )

            await agent.startup()
            result = await agent.runtime(task="finish")

        self.assertEqual(json.loads(result or "")["summary"], "done")
        self.assertEqual(handler.started, 1)
        self.assertEqual(handler.task_starts, 1)

    async def test_tool_middleware_lifecycle_and_low_risk_execution(self) -> None:
        handler = EchoHandler()
        middleware = RecordingToolMiddleware()
        agent = make_agent(
            agent_id="middleware-low-risk",
            handlers=[handler],
            middlewares=[middleware],
            model=FakeModelAdapter([]),
        )

        outcome = await agent.dispatch("echo", {"text": "allowed"})
        await agent.shutdown()

        self.assertEqual(outcome.data, {"status": "success", "text": "allowed"})
        self.assertEqual(handler.calls, 1)
        self.assertEqual(middleware.started, 1)
        self.assertEqual(middleware.stopped, 1)
        self.assertEqual(middleware.calls, [("echo", {"text": "allowed"})])

    async def test_tool_middleware_task_start_hook_runs_for_each_task(self) -> None:
        middleware = RecordingToolMiddleware()
        client = FakeModelAdapter(
            [
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-1",
                            function=FakeFunction("done", '{"summary": "first"}'),
                        )
                    ]
                ),
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-2",
                            function=FakeFunction("done", '{"summary": "second"}'),
                        )
                    ]
                ),
            ]
        )
        agent = make_agent(
            agent_id="middleware-task-hooks",
            handlers=[DoneHandler()],
            middlewares=[middleware],
            model=client,
        )

        await agent.runtime(task="first")
        await agent.runtime(task="second")

        self.assertEqual(middleware.task_starts, 2)

    async def test_middlewares_do_not_run_without_handlers(self) -> None:
        middleware = RecordingToolMiddleware()
        client = FakeModelAdapter([FakeMessage("plain chat")])
        agent = make_agent(
            agent_id="middleware-disabled",
            middlewares=[middleware],
            model=client,
        )

        result = await agent.runtime(task="hello")

        self.assertEqual(result, "plain chat")
        self.assertEqual(middleware.started, 0)
        self.assertEqual(middleware.task_starts, 0)
        self.assertEqual(middleware.calls, [])

    async def test_tool_middleware_rejection_has_distinct_run_status(self) -> None:
        handler = EchoHandler()
        middleware = RecordingToolMiddleware(
            StepOutcome(
                {
                    "status": "rejected",
                    "tool": "echo",
                    "reason": "human rejected tool execution",
                },
                control=ToolControl.REJECT,
            )
        )
        client = FakeModelAdapter(
            [
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            id="call-1",
                            function=FakeFunction("echo", '{"text": "blocked"}'),
                        )
                    ]
                )
            ]
        )
        agent = make_agent(
            agent_id="middleware-reject",
            handlers=[handler],
            middlewares=[middleware],
            model=client,
        )

        result = await agent.run(task="try echo")

        self.assertIsInstance(result, AgentRunResult)
        self.assertEqual(result.status, RunStatus.REJECTED)
        self.assertEqual(result.stop_reason, StopReason.TOOL_REJECTED)
        payload = json.loads(result.output or "")
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["tool"], "echo")
        self.assertEqual(handler.calls, 0)
        self.assertEqual(middleware.contexts[0].tool_call_id, "call-1")
        self.assertIs(middleware.contexts[0].state, agent.state)

    async def test_runtime_raises_structured_error_for_rejection(self) -> None:
        middleware = RecordingToolMiddleware(
            StepOutcome({"status": "rejected"}, control=ToolControl.REJECT)
        )
        agent = make_agent(
            agent_id="middleware-reject-compatibility",
            handlers=[EchoHandler()],
            middlewares=[middleware],
            model=FakeModelAdapter(
                [
                    FakeMessage(
                        tool_calls=[
                            FakeToolCall(
                                id="call-1",
                                function=FakeFunction("echo", '{"text": "blocked"}'),
                            )
                        ]
                    )
                ]
            ),
        )

        with self.assertRaises(AgentRunError) as raised:
            await agent.runtime(task="try echo")

        self.assertEqual(raised.exception.result.status, RunStatus.REJECTED)

    async def test_middlewares_short_circuit_in_order(self) -> None:
        first = RecordingToolMiddleware(
            StepOutcome({"status": "blocked"}, control=ToolControl.REJECT)
        )
        second = RecordingToolMiddleware()
        handler = EchoHandler()
        agent = make_agent(
            agent_id="middleware-chain",
            handlers=[handler],
            middlewares=[first, second],
            model=FakeModelAdapter([]),
        )

        outcome = await agent.dispatch("echo", {"text": "blocked"})

        self.assertEqual(outcome.data, {"status": "blocked"})
        self.assertEqual(first.calls, [("echo", {"text": "blocked"})])
        self.assertEqual(second.calls, [])
        self.assertEqual(handler.calls, 0)

    async def test_tool_middlewares_wrap_handler_in_declaration_order(self) -> None:
        events: list[str] = []

        class TracingToolMiddleware(ToolMiddleware):
            def __init__(self, name: str) -> None:
                super().__init__(name=name)

            async def __call__(
                self,
                context: ToolCallContext,
                call_next: ToolNext,
            ) -> StepOutcome:
                events.append(f"{self.name}:before")
                try:
                    return await call_next(context)
                finally:
                    events.append(f"{self.name}:after")

        agent = make_agent(
            agent_id="middleware-wrap-order",
            handlers=[EchoHandler()],
            middlewares=[
                TracingToolMiddleware("first"),
                TracingToolMiddleware("second"),
            ],
            model=FakeModelAdapter([]),
        )

        outcome = await agent.dispatch("echo", {"text": "wrapped"})

        self.assertEqual(outcome.data, {"status": "success", "text": "wrapped"})
        self.assertEqual(
            events,
            ["first:before", "second:before", "second:after", "first:after"],
        )

    async def test_outer_tool_middleware_observes_inner_short_circuit(self) -> None:
        events: list[str] = []

        class ObservingToolMiddleware(ToolMiddleware):
            async def __call__(
                self,
                context: ToolCallContext,
                call_next: ToolNext,
            ) -> StepOutcome:
                events.append("outer:before")
                try:
                    return await call_next(context)
                finally:
                    events.append("outer:after")

        handler = EchoHandler()
        blocking = RecordingToolMiddleware(
            StepOutcome({"status": "blocked"}, control=ToolControl.REJECT)
        )
        agent = make_agent(
            agent_id="middleware-observe-short-circuit",
            handlers=[handler],
            middlewares=[ObservingToolMiddleware(), blocking],
            model=FakeModelAdapter([]),
        )

        outcome = await agent.dispatch("echo", {"text": "blocked"})

        self.assertEqual(outcome.data, {"status": "blocked"})
        self.assertEqual(events, ["outer:before", "outer:after"])
        self.assertEqual(handler.calls, 0)

    async def test_outer_tool_middleware_observes_handler_exception(self) -> None:
        events: list[str] = []

        class FailingEchoHandler(EchoHandler):
            async def do_echo(
                self,
                arguments: dict[str, Any],
                *,
                cancellation: CancellationToken | None = None,
            ) -> StepOutcome:
                self.calls += 1
                raise RuntimeError("handler failed")

        class ObservingToolMiddleware(ToolMiddleware):
            async def __call__(
                self,
                context: ToolCallContext,
                call_next: ToolNext,
            ) -> StepOutcome:
                events.append("outer:before")
                try:
                    return await call_next(context)
                finally:
                    events.append("outer:after")

        handler = FailingEchoHandler()
        agent = make_agent(
            agent_id="middleware-observe-error",
            handlers=[handler],
            middlewares=[ObservingToolMiddleware()],
            model=FakeModelAdapter([]),
        )

        with self.assertRaisesRegex(RuntimeError, "handler failed"):
            await agent.dispatch("echo", {"text": "fail"})

        self.assertEqual(events, ["outer:before", "outer:after"])
        self.assertEqual(handler.calls, 1)

    async def test_started_tool_middleware_still_shuts_down_if_disabled(self) -> None:
        middleware = RecordingToolMiddleware()
        agent = make_agent(
            agent_id="middleware-toggle",
            handlers=[EchoHandler()],
            middlewares=[middleware],
            model=FakeModelAdapter([]),
        )

        await agent.startup()
        middleware.enabled = False
        await agent.shutdown()

        self.assertEqual(middleware.started, 1)
        self.assertEqual(middleware.stopped, 1)

    async def test_third_identical_tool_call_fails_before_execution(self) -> None:
        handler = EchoHandler()
        repeated_call = FakeToolCall(
            id="call",
            function=FakeFunction("echo", '{"text": "same"}'),
        )
        client = FakeModelAdapter(
            [
                FakeMessage(tool_calls=[repeated_call]),
                FakeMessage(tool_calls=[repeated_call]),
                FakeMessage(tool_calls=[repeated_call]),
            ]
        )
        agent = make_agent(
            agent_id="repeat-guard",
            handlers=[handler],
            runtime_policy=RuntimePolicy(max_steps=3),
            model=client,
        )

        with self.assertRaisesRegex(RuntimeError, "consecutive times"):
            await agent.runtime(task="repeat")

        self.assertEqual(handler.calls, 2)

    async def test_tool_mode_raises_when_finish_is_never_called(self) -> None:
        client = FakeModelAdapter([FakeMessage("not finished"), FakeMessage(None)])
        agent = make_agent(
            agent_id="step-limit",
            handlers=[DoneHandler()],
            runtime_policy=RuntimePolicy(
                max_steps=2,
                require_explicit_finish=True,
            ),
            model=client,
        )

        with self.assertRaisesRegex(RuntimeError, "did not finish within 2"):
            await agent.runtime(task="never finish")

    async def test_plain_chat_mode_does_not_start_handlers(self) -> None:
        client = FakeModelAdapter([FakeMessage("plain chat")])
        agent = make_agent(
            agent_id="plain-chat",
            model=client,
        )

        result = await agent.runtime(task="hello")

        self.assertEqual(result, "plain chat")
        self.assertIsNone(client.completions.calls[0]["tools"])

    async def test_mcp_handler_uses_the_same_handler_contract(self) -> None:
        manager = FakeMcpManager()
        handler = McpToolHandler(manager=manager)

        await handler.startup()
        outcome = await handler.dispatch("demo__lookup", {"query": "value"})
        await handler.shutdown()

        self.assertEqual(outcome.data, "mcp-result")
        self.assertEqual(
            manager.calls,
            [("demo__lookup", {"query": "value"})],
        )
        self.assertTrue(manager.started)
        self.assertTrue(manager.stopped)

    async def test_mcp_handler_requires_one_explicit_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "config_path is required"):
            McpToolHandler()
        with self.assertRaisesRegex(ValueError, "either config_path or manager"):
            McpToolHandler("mcp.json", manager=FakeMcpManager())


if __name__ == "__main__":
    unittest.main()
