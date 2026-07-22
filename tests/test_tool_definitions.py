import unittest
from types import SimpleNamespace
from typing import Any, cast

from simagentplg import (
    AgentContextBuilder,
    AgentState,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    ContextBuildResult,
    MethodToolHandler,
    ModelAdapter,
    ModelConfig,
    OpenAIModelAdapter,
    StepOutcome,
    ToolDefinition,
    ToolDefinitionError,
    ToolEffect,
)

LEGACY_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_value",
        "description": "Write one value.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}


class SequenceModel(ModelAdapter):
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = list(responses)

    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        return self.responses.pop(0)


class MixedDefinitionHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__(
            (
                ToolDefinition(
                    name="lookup",
                    description="Look up one value.",
                    parameters={
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                    effect=ToolEffect.READ_ONLY,
                    strict=True,
                ),
                LEGACY_WRITE_TOOL,
            )
        )

    async def do_lookup(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome({"key": arguments["key"]})

    async def do_write_value(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome({"value": arguments["value"]})


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


class ToolDefinitionTests(unittest.IsolatedAsyncioTestCase):
    def test_definition_serializes_openai_fields_but_not_core_effect(self) -> None:
        parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }
        definition = ToolDefinition(
            name="search",
            description="Search documents.",
            parameters=parameters,
            effect=ToolEffect.READ_ONLY,
            strict=True,
        )
        parameters["properties"]["query"]["type"] = "integer"

        serialized = definition.to_openai_tool()

        self.assertEqual(
            serialized,
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                    "strict": True,
                },
            },
        )
        self.assertNotIn("effect", serialized["function"])
        serialized["function"]["parameters"]["properties"]["query"]["type"] = "boolean"
        self.assertEqual(
            definition.to_openai_tool()["function"]["parameters"]["properties"][
                "query"
            ]["type"],
            "string",
        )

    def test_legacy_dictionary_round_trips_supported_openai_fields(self) -> None:
        legacy = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup.",
                "parameters": {"type": "object"},
                "strict": False,
            },
        }

        definition = ToolDefinition.from_openai_tool(
            legacy,
            effect=ToolEffect.READ_ONLY,
        )

        self.assertEqual(definition.name, "lookup")
        self.assertEqual(definition.effect, ToolEffect.READ_ONLY)
        self.assertEqual(definition.to_openai_tool(), legacy)

    def test_optional_openai_fields_remain_omitted(self) -> None:
        definition = ToolDefinition.from_openai_tool(
            {"type": "function", "function": {"name": "minimal"}}
        )

        self.assertEqual(
            definition.to_openai_tool(),
            {"type": "function", "function": {"name": "minimal"}},
        )

    def test_definition_rejects_invalid_core_and_openai_values(self) -> None:
        cases = [
            lambda: ToolDefinition(name=""),
            lambda: ToolDefinition(name="invalid name"),
            lambda: ToolDefinition(name="x" * 65),
            lambda: ToolDefinition(name="tool", description=1),
            lambda: ToolDefinition(name="tool", parameters=[]),
            lambda: ToolDefinition(name="tool", parameters={"bad": {1, 2}}),
            lambda: ToolDefinition(name="tool", parameters={"bad": float("nan")}),
            lambda: ToolDefinition(name="tool", effect="read_only"),
            lambda: ToolDefinition(name="tool", strict="yes"),
            lambda: ToolDefinition.from_openai_tool({"type": "custom"}),
            lambda: ToolDefinition.from_openai_tool(
                {"type": "function", "function": []}
            ),
            lambda: ToolDefinition.from_openai_tool(
                {"type": "function", "function": {}}
            ),
        ]
        for create in cases:
            with self.subTest(create=create), self.assertRaises(ToolDefinitionError):
                create()

    def test_parameters_are_deeply_immutable(self) -> None:
        definition = ToolDefinition(
            name="immutable",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        )
        assert definition.parameters is not None

        with self.assertRaises(TypeError):
            definition.parameters["type"] = "array"  # type: ignore[index]
        properties = definition.parameters["properties"]
        with self.assertRaises(TypeError):
            properties["value"]["type"] = "number"

    def test_method_handler_accepts_canonical_and_legacy_definitions(self) -> None:
        handler = MixedDefinitionHandler()

        self.assertEqual(handler.tool_names, ("lookup", "write_value"))
        self.assertTrue(
            all(isinstance(tool, ToolDefinition) for tool in handler.tool_definitions)
        )
        self.assertEqual(
            [tool.effect for tool in handler.tool_definitions],
            [ToolEffect.READ_ONLY, ToolEffect.SIDE_EFFECTING],
        )
        self.assertTrue(handler.tools[0]["function"]["strict"])

        detached = handler.tools
        detached[0]["function"]["name"] = "mutated"
        self.assertEqual(handler.tool_definitions[0].name, "lookup")

    def test_legacy_effect_mapping_remains_compatible(self) -> None:
        handler = MethodToolHandler(
            (LEGACY_WRITE_TOOL,),
            tool_effects={"write_value": ToolEffect.READ_ONLY},
        )

        self.assertEqual(
            handler.tool_definitions[0].effect,
            ToolEffect.READ_ONLY,
        )

    def test_mixed_duplicate_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ToolDefinitionError, "duplicate tool names"):
            MethodToolHandler(
                (
                    ToolDefinition(name="duplicate"),
                    {
                        "type": "function",
                        "function": {"name": "duplicate"},
                    },
                )
            )

    def test_context_builder_keeps_canonical_source_and_legacy_view(self) -> None:
        definition = ToolDefinition(
            name="lookup",
            parameters={"type": "object"},
            effect=ToolEffect.READ_ONLY,
        )

        context = AgentContextBuilder().build(
            AgentState(messages=[{"role": "user", "content": "hello"}]),
            tools=(definition,),
        )

        self.assertEqual(context.tool_definitions, (definition,))
        self.assertEqual(
            context.tools,
            (
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
            ),
        )

    async def test_agent_routes_canonical_definition_and_exposes_legacy_view(
        self,
    ) -> None:
        handler = MixedDefinitionHandler()
        agent = BaseAgent(
            SequenceModel([AssistantMessage(content="done")]),
            agent_id="canonical-agent",
            handlers=[handler],
        )

        result = await agent.run(task="inspect tools")

        self.assertEqual(result.output, "done")
        self.assertEqual(
            [tool.name for tool in agent.tool_definitions],
            ["lookup", "write_value"],
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in agent.tools],
            ["lookup", "write_value"],
        )

    async def test_openai_adapter_prefers_canonical_context_definitions(self) -> None:
        canonical = ToolDefinition(
            name="canonical",
            parameters={"type": "object"},
            strict=True,
        )
        context = ContextBuildResult(
            agent_messages=({"role": "user", "content": "hello"},),
            llm_messages=({"role": "user", "content": "hello"},),
            tools=(
                {
                    "type": "function",
                    "function": {"name": "legacy-view"},
                },
            ),
            tool_definitions=(canonical,),
        )
        client = FakeClient()
        adapter = OpenAIModelAdapter(
            ModelConfig(
                model="test",
                api_key="test",
                base_url="https://example.invalid",
            ),
            client=cast(Any, client),
        )

        response = await adapter.complete(context)

        self.assertEqual(response.content, "done")
        self.assertEqual(
            client.chat.completions.calls[0]["tools"],
            (
                {
                    "type": "function",
                    "function": {
                        "name": "canonical",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
