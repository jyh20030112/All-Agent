from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from ejagent import (
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    MethodToolHandler,
    ModelAdapter,
    ModelToolCall,
    RuleBasedToolPolicy,
    RunStatus,
    RuntimePolicy,
    StepOutcome,
    StopReason,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCallContext,
    ToolDefinition,
    ToolEffect,
    ToolExecutionPolicy,
    ToolPolicyAction,
    ToolPolicyDecision,
    ToolPolicyMiddleware,
    ToolPolicyRule,
)
from ejagent.agent.state import AgentState

READ_TOOL = ToolDefinition(
    name="read",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    effect=ToolEffect.READ_ONLY,
)
WRITE_TOOL = ToolDefinition(
    name="write",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)


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


class RecordingHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((READ_TOOL, WRITE_TOOL))
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def do_read(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken,
    ) -> StepOutcome:
        self.calls.append(("read", arguments))
        return StepOutcome({"status": "success", "tool": "read"})

    async def do_write(
        self,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationToken,
    ) -> StepOutcome:
        self.calls.append(("write", arguments))
        return StepOutcome({"status": "success", "tool": "write"})


class RecordingApprover:
    def __init__(self, decision: ToolApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[ToolApprovalRequest] = []

    async def approve(self, request: ToolApprovalRequest) -> ToolApprovalDecision:
        self.requests.append(request)
        return self.decision


class RaisingApprover:
    async def approve(self, request: ToolApprovalRequest) -> ToolApprovalDecision:
        raise RuntimeError("approval service unavailable")


class LifecyclePolicy(ToolExecutionPolicy):
    def __init__(self, decision: ToolPolicyDecision) -> None:
        self.decision = decision
        self.started = 0
        self.stopped = 0
        self.task_starts = 0
        self.contexts: list[ToolCallContext] = []

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1

    async def on_task_start(self) -> None:
        self.task_starts += 1

    async def evaluate(self, context: ToolCallContext) -> ToolPolicyDecision:
        self.contexts.append(context)
        return self.decision


class RaisingPolicy(ToolExecutionPolicy):
    async def evaluate(self, context: ToolCallContext) -> ToolPolicyDecision:
        raise RuntimeError("policy backend unavailable")


def call(call_id: str, name: str, path: str) -> ModelToolCall:
    return ModelToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps({"path": path}),
    )


def context(
    name: str = "read",
    *,
    effect: ToolEffect = ToolEffect.READ_ONLY,
) -> ToolCallContext:
    state = AgentState()
    state.reset([])
    return ToolCallContext(
        state=state,
        tool_name=name,
        arguments={"path": "/workspace/file.txt"},
        tool_call_id="call-1",
        tool_definition=ToolDefinition(name=name, effect=effect),
    )


class ToolPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_deny_short_circuits_handler_and_agent_run(self) -> None:
        handler = RecordingHandler()
        policy = RuleBasedToolPolicy(())
        agent = BaseAgent(
            SequenceModel(
                [AssistantMessage(tool_calls=(call("one", "write", "/tmp/a"),))]
            ),
            agent_id="policy-default-deny",
            handlers=[handler],
            middlewares=[ToolPolicyMiddleware(policy)],
        )

        result = await agent.run(task="write a file")

        self.assertEqual(result.status, RunStatus.REJECTED)
        self.assertEqual(result.stop_reason, StopReason.TOOL_REJECTED)
        self.assertEqual(handler.calls, [])
        payload = json.loads(result.output or "")
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["tool"], "write")
        self.assertNotIn("arguments", payload)

    async def test_effect_rule_uses_effective_canonical_definition(self) -> None:
        handler = RecordingHandler()
        policy = LifecyclePolicy(ToolPolicyDecision(ToolPolicyAction.ALLOW))
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="policy-definition",
            handlers=[handler],
            middlewares=[ToolPolicyMiddleware(policy)],
        )

        outcome = await agent.dispatch("read", {"path": "/workspace/a"})

        self.assertEqual(outcome.data["status"], "success")
        definition = policy.contexts[0].tool_definition
        self.assertIsNotNone(definition)
        assert definition is not None
        self.assertEqual(definition.name, "read")
        self.assertIs(definition.effect, ToolEffect.READ_ONLY)

    async def test_ordered_predicate_rule_denies_matching_arguments(self) -> None:
        policy = RuleBasedToolPolicy(
            (
                ToolPolicyRule(
                    rule_id="deny-private",
                    action=ToolPolicyAction.DENY,
                    tool_names=frozenset({"read"}),
                    when=lambda item: str(item.arguments["path"]).startswith(
                        "/private"
                    ),
                    reason="private paths are forbidden",
                ),
                ToolPolicyRule(
                    rule_id="allow-read",
                    action=ToolPolicyAction.ALLOW,
                    tool_names=frozenset({"read"}),
                ),
            )
        )

        denied = await policy.evaluate(
            ToolCallContext(
                state=AgentState(),
                tool_name="read",
                arguments={"path": "/private/secret"},
                tool_definition=READ_TOOL,
            )
        )
        allowed = await policy.evaluate(context())

        self.assertIs(denied.action, ToolPolicyAction.DENY)
        self.assertEqual(denied.rule_id, "deny-private")
        self.assertIs(allowed.action, ToolPolicyAction.ALLOW)
        self.assertEqual(allowed.rule_id, "allow-read")

    async def test_effect_selector_can_require_side_effect_approval(self) -> None:
        policy = RuleBasedToolPolicy(
            (
                ToolPolicyRule(
                    rule_id="approve-writes",
                    action=ToolPolicyAction.REQUIRE_APPROVAL,
                    effects=frozenset({ToolEffect.SIDE_EFFECTING}),
                ),
                ToolPolicyRule(
                    rule_id="allow-reads",
                    action=ToolPolicyAction.ALLOW,
                    effects=frozenset({ToolEffect.READ_ONLY}),
                ),
            )
        )

        read = await policy.evaluate(context())
        write = await policy.evaluate(
            context("write", effect=ToolEffect.SIDE_EFFECTING)
        )

        self.assertIs(read.action, ToolPolicyAction.ALLOW)
        self.assertIs(write.action, ToolPolicyAction.REQUIRE_APPROVAL)

    async def test_required_approval_without_approver_fails_closed(self) -> None:
        handler = RecordingHandler()
        policy = LifecyclePolicy(
            ToolPolicyDecision(
                ToolPolicyAction.REQUIRE_APPROVAL,
                rule_id="approve-write",
            )
        )
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="policy-no-approver",
            handlers=[handler],
            middlewares=[ToolPolicyMiddleware(policy)],
        )

        outcome = await agent.dispatch("write", {"path": "/tmp/a"})

        self.assertEqual(outcome.control.value, "reject")
        self.assertEqual(outcome.data["rule_id"], "approve-write")
        self.assertEqual(handler.calls, [])

    async def test_approved_call_executes_and_request_is_immutable(self) -> None:
        handler = RecordingHandler()
        policy = LifecyclePolicy(
            ToolPolicyDecision(
                ToolPolicyAction.REQUIRE_APPROVAL,
                reason="write changes external state",
                rule_id="approve-write",
            )
        )
        approver = RecordingApprover(ToolApprovalDecision(True, "approved"))
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="policy-approved",
            handlers=[handler],
            middlewares=[ToolPolicyMiddleware(policy, approver=approver)],
        )

        outcome = await agent.dispatch("write", {"path": "/tmp/a"})

        self.assertEqual(outcome.data["status"], "success")
        self.assertEqual(handler.calls, [("write", {"path": "/tmp/a"})])
        request = approver.requests[0]
        self.assertEqual(request.tool_name, "write")
        self.assertEqual(request.arguments["path"], "/tmp/a")
        self.assertEqual(request.rule_id, "approve-write")
        with self.assertRaises(TypeError):
            request.arguments["path"] = "/tmp/changed"  # type: ignore[index]

    async def test_denied_approval_uses_approver_reason(self) -> None:
        handler = RecordingHandler()
        policy = LifecyclePolicy(
            ToolPolicyDecision(
                ToolPolicyAction.REQUIRE_APPROVAL,
                reason="approval required",
                rule_id="approve-write",
            )
        )
        approver = RecordingApprover(
            ToolApprovalDecision(False, "operator denied the write")
        )
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="policy-approval-denied",
            handlers=[handler],
            middlewares=[ToolPolicyMiddleware(policy, approver=approver)],
        )

        outcome = await agent.dispatch("write", {"path": "/tmp/a"})

        self.assertEqual(outcome.control.value, "reject")
        self.assertEqual(outcome.data["reason"], "operator denied the write")
        self.assertEqual(handler.calls, [])

    async def test_approval_exception_fails_closed(self) -> None:
        handler = RecordingHandler()
        policy = LifecyclePolicy(
            ToolPolicyDecision(
                ToolPolicyAction.REQUIRE_APPROVAL,
                rule_id="approve-write",
            )
        )
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="policy-approval-error",
            handlers=[handler],
            middlewares=[ToolPolicyMiddleware(policy, approver=RaisingApprover())],
        )

        with self.assertLogs("ejagent.middleware.policy", level="ERROR"):
            outcome = await agent.dispatch("write", {"path": "/tmp/a"})

        self.assertEqual(outcome.control.value, "reject")
        self.assertEqual(outcome.data["reason"], "tool approval failed")
        self.assertEqual(handler.calls, [])

    async def test_policy_exception_fails_closed(self) -> None:
        handler = RecordingHandler()
        agent = BaseAgent(
            SequenceModel([]),
            agent_id="policy-error",
            handlers=[handler],
            middlewares=[ToolPolicyMiddleware(RaisingPolicy())],
        )

        with self.assertLogs("ejagent.middleware.policy", level="ERROR"):
            outcome = await agent.dispatch("read", {"path": "/tmp/a"})

        self.assertEqual(outcome.control.value, "reject")
        self.assertEqual(outcome.data["reason"], "tool policy evaluation failed")
        self.assertEqual(handler.calls, [])

    async def test_policy_lifecycle_delegates_through_middleware(self) -> None:
        policy = LifecyclePolicy(ToolPolicyDecision(ToolPolicyAction.ALLOW))
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(content="first"),
                    AssistantMessage(content="second"),
                ]
            ),
            agent_id="policy-lifecycle",
            handlers=[RecordingHandler()],
            middlewares=[ToolPolicyMiddleware(policy)],
        )

        await agent.run(task="first")
        await agent.run(task="second")
        await agent.shutdown()

        self.assertEqual(policy.started, 1)
        self.assertEqual(policy.task_starts, 2)
        self.assertEqual(policy.stopped, 1)

    async def test_call_limit_reservation_is_atomic_and_resets_per_run(self) -> None:
        policy = RuleBasedToolPolicy(
            (
                ToolPolicyRule(
                    rule_id="read-budget",
                    action=ToolPolicyAction.ALLOW,
                    tool_names=frozenset({"read"}),
                    max_calls_per_run=3,
                ),
            )
        )
        await policy.on_task_start()

        decisions = await asyncio.gather(
            *(policy.evaluate(context()) for _ in range(20))
        )

        self.assertEqual(
            sum(item.action is ToolPolicyAction.ALLOW for item in decisions),
            3,
        )
        self.assertEqual(
            sum(item.action is ToolPolicyAction.DENY for item in decisions),
            17,
        )
        self.assertTrue(
            all(
                item.rule_id == "read-budget"
                for item in decisions
                if item.action is ToolPolicyAction.DENY
            )
        )

        await policy.on_task_start()
        reset_decision = await policy.evaluate(context())

        self.assertIs(reset_decision.action, ToolPolicyAction.ALLOW)

    async def test_parallel_tool_calls_cannot_overrun_policy_limit(self) -> None:
        handler = RecordingHandler()
        policy = RuleBasedToolPolicy(
            (
                ToolPolicyRule(
                    rule_id="parallel-read-budget",
                    action=ToolPolicyAction.ALLOW,
                    tool_names=frozenset({"read"}),
                    max_calls_per_run=2,
                ),
            )
        )
        agent = BaseAgent(
            SequenceModel(
                [
                    AssistantMessage(
                        tool_calls=tuple(
                            call(str(index), "read", f"/workspace/{index}")
                            for index in range(4)
                        )
                    )
                ]
            ),
            agent_id="policy-parallel-limit",
            handlers=[handler],
            middlewares=[ToolPolicyMiddleware(policy)],
            runtime_policy=RuntimePolicy(parallel_tool_calls=True),
        )

        result = await agent.run(task="read four files")

        self.assertEqual(result.status, RunStatus.REJECTED)
        self.assertEqual(result.stop_reason, StopReason.TOOL_REJECTED)
        self.assertEqual(len(handler.calls), 2)

    async def test_invalid_rule_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RuleBasedToolPolicy(
                (
                    ToolPolicyRule("duplicate", ToolPolicyAction.ALLOW),
                    ToolPolicyRule("duplicate", ToolPolicyAction.DENY),
                )
            )
        with self.assertRaises(ValueError):
            ToolPolicyRule(
                "deny-limit",
                ToolPolicyAction.DENY,
                max_calls_per_run=1,
            )


if __name__ == "__main__":
    unittest.main()
