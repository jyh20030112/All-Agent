from __future__ import annotations

import json
import unittest
from typing import Any

from simagentplg import (
    AgentRunResult,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    MemorySessionStorage,
    MethodToolHandler,
    ModelAdapter,
    ModelToolCall,
    RuleBasedToolPolicy,
    RunStatus,
    RuntimePolicy,
    SessionRecorder,
    StepOutcome,
    ToolDefinition,
    ToolEffect,
    ToolExecutionPolicy,
    ToolPolicyAction,
    ToolPolicyDecision,
    ToolPolicyMiddleware,
    ToolPolicyRule,
    ToolSchemaConfigurationError,
    ToolSchemaValidationMiddleware,
)

TRANSFER_TOOL = ToolDefinition(
    name="transfer",
    parameters={
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "amount": {"type": "number", "minimum": 0},
            "currency": {"type": "string", "enum": ["USD", "CNY"]},
        },
        "required": ["to", "amount", "currency"],
        "additionalProperties": False,
    },
)
INSPECT_TOOL = ToolDefinition(
    name="inspect_items",
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    effect=ToolEffect.READ_ONLY,
)
PING_TOOL = ToolDefinition(name="ping", effect=ToolEffect.READ_ONLY)


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


class ValidationHandler(MethodToolHandler):
    def __init__(
        self,
        tools: tuple[ToolDefinition | dict[str, Any], ...] = (
            TRANSFER_TOOL,
            INSPECT_TOOL,
            PING_TOOL,
        ),
    ) -> None:
        super().__init__(tools)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.started = 0
        self.stopped = 0

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1

    async def do_transfer(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken,
    ) -> StepOutcome:
        self.calls.append(("transfer", arguments))
        return StepOutcome({"status": "success", "tool": "transfer"})

    async def do_inspect_items(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken,
    ) -> StepOutcome:
        self.calls.append(("inspect_items", arguments))
        return StepOutcome({"status": "success", "tool": "inspect_items"})

    async def do_ping(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken,
    ) -> StepOutcome:
        self.calls.append(("ping", arguments))
        return StepOutcome({"status": "success", "tool": "ping"})


class CountingPolicy(ToolExecutionPolicy):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def evaluate(self, context: Any) -> ToolPolicyDecision:
        self.calls.append(context.tool_name)
        return ToolPolicyDecision(ToolPolicyAction.ALLOW)


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> ModelToolCall:
    return ModelToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


class ToolSchemaValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_arguments_reach_handler(self) -> None:
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-valid",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )
        arguments = {"to": "alice", "amount": 25.5, "currency": "USD"}

        outcome = await agent.dispatch("transfer", arguments)

        self.assertEqual(outcome.data["status"], "success")
        self.assertEqual(handler.calls, [("transfer", arguments)])

    async def test_type_error_is_safe_and_short_circuits_handler(self) -> None:
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-type-error",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )
        secret = "secret-account-token"

        outcome = await agent.dispatch(
            "transfer",
            {"to": "alice", "amount": secret, "currency": "USD"},
        )

        self.assertEqual(outcome.control.value, "continue")
        self.assertEqual(outcome.data["code"], "invalid_tool_arguments")
        self.assertEqual(outcome.data["errors"][0]["path"], "/amount")
        self.assertEqual(outcome.data["errors"][0]["keyword"], "type")
        self.assertNotIn(secret, json.dumps(outcome.data))
        self.assertEqual(handler.calls, [])

    async def test_required_error_points_to_missing_property(self) -> None:
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-required",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )

        outcome = await agent.dispatch(
            "transfer",
            {"amount": 10, "currency": "CNY"},
        )

        self.assertEqual(outcome.data["errors"][0]["path"], "/to")
        self.assertEqual(outcome.data["errors"][0]["keyword"], "required")
        self.assertEqual(handler.calls, [])

    async def test_multiple_required_errors_point_to_distinct_properties(
        self,
    ) -> None:
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-required-many",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )

        outcome = await agent.dispatch("transfer", {})

        paths = {error["path"] for error in outcome.data["errors"]}
        self.assertEqual(paths, {"/amount", "/currency", "/to"})
        self.assertEqual(handler.calls, [])

    async def test_nested_array_error_uses_json_pointer(self) -> None:
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-nested",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )

        outcome = await agent.dispatch(
            "inspect_items",
            {"items": [{"count": "many"}]},
        )

        self.assertEqual(outcome.data["errors"][0]["path"], "/items/0/count")
        self.assertEqual(outcome.data["errors"][0]["keyword"], "type")
        self.assertEqual(handler.calls, [])

    async def test_enum_and_additional_property_errors_do_not_echo_values(
        self,
    ) -> None:
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-multiple-errors",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )
        secret = "secret-field-value"

        outcome = await agent.dispatch(
            "transfer",
            {
                "to": "alice",
                "amount": 10,
                "currency": "EUR",
                "unexpected": secret,
            },
        )

        keywords = {error["keyword"] for error in outcome.data["errors"]}
        self.assertEqual(keywords, {"additionalProperties", "enum"})
        self.assertNotIn("EUR", json.dumps(outcome.data))
        self.assertNotIn(secret, json.dumps(outcome.data))
        self.assertEqual(handler.calls, [])

    async def test_error_output_is_bounded(self) -> None:
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-error-limit",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware(max_errors=1)],
        )

        outcome = await agent.dispatch("transfer", {})

        self.assertEqual(len(outcome.data["errors"]), 1)
        self.assertTrue(outcome.data["truncated"])
        self.assertEqual(handler.calls, [])

    async def test_tool_without_parameters_schema_passes_through(self) -> None:
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-absent",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )

        outcome = await agent.dispatch("ping", {"anything": "is accepted"})

        self.assertEqual(outcome.data["status"], "success")
        self.assertEqual(
            handler.calls,
            [("ping", {"anything": "is accepted"})],
        )

    async def test_legacy_openai_definition_is_normalized_and_validated(
        self,
    ) -> None:
        legacy_tool = {
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        handler = ValidationHandler((legacy_tool,))
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-legacy",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )

        outcome = await agent.dispatch("ping", {"value": "wrong"})

        self.assertEqual(outcome.data["code"], "invalid_tool_arguments")
        self.assertEqual(handler.calls, [])

    async def test_invalid_schema_fails_agent_startup_and_rolls_back_handler(
        self,
    ) -> None:
        invalid_tool = ToolDefinition(
            name="ping",
            parameters={"type": "not-a-json-schema-type"},
        )
        handler = ValidationHandler((invalid_tool,))
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-invalid-config",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )

        with self.assertRaises(ToolSchemaConfigurationError) as raised:
            await agent.startup()

        self.assertEqual(raised.exception.tool_name, "ping")
        self.assertEqual(handler.started, 1)
        self.assertEqual(handler.stopped, 1)

    async def test_disabled_validation_preserves_existing_schema_behavior(
        self,
    ) -> None:
        invalid_tool = ToolDefinition(
            name="ping",
            parameters={"type": "not-a-json-schema-type"},
        )
        handler = ValidationHandler((invalid_tool,))
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-disabled",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware(enabled=False)],
        )

        outcome = await agent.dispatch("ping", {"legacy": "arguments"})

        self.assertEqual(outcome.data["status"], "success")
        self.assertEqual(
            handler.calls,
            [("ping", {"legacy": "arguments"})],
        )

    async def test_remote_schema_reference_is_not_retrieved(self) -> None:
        remote_tool = ToolDefinition(
            name="ping",
            parameters={"$ref": "https://schemas.example.invalid/tool.json"},
        )
        handler = ValidationHandler((remote_tool,))
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-no-remote",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )

        with self.assertRaises(ToolSchemaConfigurationError) as raised:
            await agent.dispatch("ping", {})

        self.assertEqual(raised.exception.tool_name, "ping")
        self.assertEqual(handler.calls, [])

    async def test_local_schema_reference_is_supported(self) -> None:
        referenced_tool = ToolDefinition(
            name="ping",
            parameters={
                "$defs": {"count": {"type": "integer", "minimum": 1}},
                "type": "object",
                "properties": {"count": {"$ref": "#/$defs/count"}},
                "required": ["count"],
                "additionalProperties": False,
            },
        )
        handler = ValidationHandler((referenced_tool,))
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-local-ref",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
        )

        invalid = await agent.dispatch("ping", {"count": 0})
        valid = await agent.dispatch("ping", {"count": 2})

        self.assertEqual(invalid.data["errors"][0]["keyword"], "minimum")
        self.assertEqual(valid.data["status"], "success")
        self.assertEqual(handler.calls, [("ping", {"count": 2})])

    async def test_validation_precedes_policy_and_approval_path(self) -> None:
        handler = ValidationHandler()
        policy = CountingPolicy()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="schema-before-policy",
            handlers=[handler],
            middlewares=[
                ToolSchemaValidationMiddleware(),
                ToolPolicyMiddleware(policy),
            ],
        )

        invalid = await agent.dispatch(
            "transfer",
            {"to": "alice", "amount": "wrong", "currency": "USD"},
        )
        valid = await agent.dispatch(
            "transfer",
            {"to": "alice", "amount": 10, "currency": "USD"},
        )

        self.assertEqual(invalid.data["code"], "invalid_tool_arguments")
        self.assertEqual(valid.data["status"], "success")
        self.assertEqual(policy.calls, ["transfer"])
        self.assertEqual(len(handler.calls), 1)

    async def test_parallel_calls_validate_independently_before_policy(self) -> None:
        handler = ValidationHandler()
        policy = RuleBasedToolPolicy(
            (
                ToolPolicyRule(
                    rule_id="allow-inspect",
                    action=ToolPolicyAction.ALLOW,
                    tool_names=frozenset({"inspect_items"}),
                    max_calls_per_run=1,
                ),
            )
        )
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=(
                            tool_call(
                                "valid",
                                "inspect_items",
                                {"items": [{"count": 1}]},
                            ),
                            tool_call(
                                "invalid",
                                "inspect_items",
                                {"items": [{"count": "wrong"}]},
                            ),
                        )
                    ),
                    AssistantMessage(content="done"),
                ]
            ),
            agent_id="schema-parallel",
            handlers=[handler],
            middlewares=[
                ToolSchemaValidationMiddleware(),
                ToolPolicyMiddleware(policy),
            ],
            runtime_policy=RuntimePolicy(parallel_tool_calls=True),
        )

        result = await agent.run(task="inspect two inputs")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(len(handler.calls), 1)
        tool_messages = [
            message for message in agent.messages if message.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 2)
        self.assertIn("invalid_tool_arguments", tool_messages[1]["content"])

    async def test_validation_error_is_persisted_as_normal_tool_result(self) -> None:
        storage = MemorySessionStorage()
        recorder = SessionRecorder(session_id="schema-session", storage=storage)
        handler = ValidationHandler()
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=(
                            tool_call(
                                "invalid",
                                "transfer",
                                {
                                    "to": "alice",
                                    "amount": "wrong",
                                    "currency": "USD",
                                },
                            ),
                        )
                    ),
                    AssistantMessage(content="corrected later"),
                ]
            ),
            agent_id="schema-session-agent",
            handlers=[handler],
            middlewares=[ToolSchemaValidationMiddleware()],
            event_sink=recorder,
        )

        result = await agent.run(task="transfer")
        restored = await recorder.load()

        self.assertIsInstance(result, AgentRunResult)
        self.assertIsNotNone(restored)
        assert restored is not None
        tool_messages = [
            message for message in restored.messages if message.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("invalid_tool_arguments", tool_messages[0]["content"])
        self.assertEqual(handler.calls, [])

    async def test_constructor_rejects_invalid_error_limit(self) -> None:
        with self.assertRaises(TypeError):
            ToolSchemaValidationMiddleware(max_errors=True)
        with self.assertRaises(ValueError):
            ToolSchemaValidationMiddleware(max_errors=0)


if __name__ == "__main__":
    unittest.main()
