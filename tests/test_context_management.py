import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from ejagent import (
    AgentContextBuilder,
    AgentEvent,
    AgentFinished,
    AgentStarted,
    AgentState,
    AssistantMessage,
    BaseAgent,
    CancellationToken,
    CompactionDecisionReason,
    CompactionPolicy,
    ContextBudget,
    ContextPressureEvaluated,
    ContextUsageSource,
    HeuristicMessageTokenEstimator,
    MessageCompleted,
    ModelAdapter,
    TurnCompleted,
    TurnStarted,
    estimate_context_usage,
    prepare_compaction,
)


class MarkerTokenEstimator:
    def estimate_message(self, message: Mapping[str, Any]) -> int:
        return int(message.get("_tokens", 0))

    def estimate_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        return sum(int(tool.get("_tokens", 0)) for tool in tools)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


class TextModel(ModelAdapter):
    async def complete(
        self,
        context: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        return AssistantMessage(content="done")


class ContextManagementTests(unittest.IsolatedAsyncioTestCase):
    def test_context_builder_strips_current_and_legacy_internal_metadata(self) -> None:
        state = AgentState(
            messages=[
                {
                    "role": "system",
                    "content": "summary",
                    "_ejagent_summary": {"content": "current"},
                    "_simagentplg_summary": {"content": "legacy"},
                }
            ]
        )

        context = AgentContextBuilder().build(state)

        self.assertEqual(
            context.agent_messages[0]["_ejagent_summary"], {"content": "current"}
        )
        self.assertEqual(
            context.agent_messages[0]["_simagentplg_summary"],
            {"content": "legacy"},
        )
        self.assertNotIn("_ejagent_summary", context.llm_messages[0])
        self.assertNotIn("_simagentplg_summary", context.llm_messages[0])

    def test_heuristic_is_utf8_aware_and_ignores_usage_metadata(self) -> None:
        estimator = HeuristicMessageTokenEstimator()
        english = {"role": "user", "content": "a" * 40}
        chinese = {"role": "user", "content": "你" * 40}
        with_usage = {
            **english,
            "usage": {
                "input_tokens": 9999,
                "output_tokens": 9999,
                "total_tokens": 19998,
            },
        }
        with_internal_summary = {
            **english,
            "_ejagent_summary": {
                "content": "internal metadata must not be counted" * 100,
            },
        }
        with_legacy_internal_summary = {
            **english,
            "_simagentplg_summary": {
                "content": "legacy internal metadata must not be counted" * 100,
            },
        }

        self.assertGreater(
            estimator.estimate_message(chinese),
            estimator.estimate_message(english),
        )
        self.assertEqual(
            estimator.estimate_message(english),
            estimator.estimate_message(with_usage),
        )
        self.assertEqual(
            estimator.estimate_message(english),
            estimator.estimate_message(with_internal_summary),
        )
        self.assertEqual(
            estimator.estimate_message(english),
            estimator.estimate_message(with_legacy_internal_summary),
        )

    def test_estimate_without_usage_includes_messages_and_tools(self) -> None:
        estimate = estimate_context_usage(
            [
                {"role": "system", "_tokens": 10},
                {"role": "user", "_tokens": 20},
            ],
            tools=[{"_tokens": 30}],
            estimator=MarkerTokenEstimator(),
        )

        self.assertEqual(estimate.reported_tokens, 0)
        self.assertEqual(estimate.trailing_tokens, 60)
        self.assertEqual(estimate.heuristic_tokens, 60)
        self.assertEqual(estimate.total_tokens, 60)
        self.assertIsNone(estimate.last_usage_index)
        self.assertEqual(estimate.source, ContextUsageSource.ESTIMATED)

    def test_latest_usage_is_combined_with_only_trailing_messages(self) -> None:
        estimate = estimate_context_usage(
            [
                {"role": "system", "_tokens": 10},
                {
                    "role": "assistant",
                    "_tokens": 10,
                    "usage": {"total_tokens": 500},
                },
                {"role": "tool", "_tokens": 30},
                {"role": "user", "_tokens": 20},
            ],
            tools=[{"_tokens": 40}],
            estimator=MarkerTokenEstimator(),
        )

        self.assertEqual(estimate.reported_tokens, 500)
        self.assertEqual(estimate.trailing_tokens, 50)
        self.assertEqual(estimate.heuristic_tokens, 110)
        self.assertEqual(estimate.total_tokens, 550)
        self.assertEqual(estimate.last_usage_index, 1)
        self.assertEqual(estimate.source, ContextUsageSource.MIXED)

    def test_full_heuristic_is_a_lower_bound_for_changed_context(self) -> None:
        estimate = estimate_context_usage(
            [
                {
                    "role": "assistant",
                    "_tokens": 200,
                    "usage": {"total_tokens": 50},
                }
            ],
            estimator=MarkerTokenEstimator(),
        )

        self.assertEqual(estimate.usage_based_tokens, 50)
        self.assertEqual(estimate.heuristic_tokens, 200)
        self.assertEqual(estimate.total_tokens, 200)
        self.assertEqual(estimate.source, ContextUsageSource.MIXED)

    def test_policy_is_independent_and_reports_its_reason(self) -> None:
        budget = ContextBudget(
            context_window=100,
            reserve_tokens=20,
            keep_recent_tokens=30,
        )
        policy = CompactionPolicy(budget)
        estimate = estimate_context_usage(
            [{"role": "user", "_tokens": 80}],
            estimator=MarkerTokenEstimator(),
        )

        decision = policy.evaluate(estimate)

        self.assertTrue(decision.should_compact)
        self.assertEqual(decision.threshold_tokens, 80)
        self.assertEqual(decision.pressure_ratio, 1.0)
        self.assertEqual(
            decision.reason,
            CompactionDecisionReason.THRESHOLD_REACHED,
        )

        disabled = CompactionPolicy(budget, enabled=False).evaluate(estimate)
        self.assertFalse(disabled.should_compact)
        self.assertEqual(disabled.reason, CompactionDecisionReason.DISABLED)

    def test_budget_rejects_impossible_reservations(self) -> None:
        with self.assertRaisesRegex(ValueError, "less than context_window"):
            ContextBudget(100, 100, 10)
        with self.assertRaisesRegex(ValueError, "context threshold"):
            ContextBudget(100, 20, 81)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            ContextBudget(100, 20, 0)

    def test_preparation_summarizes_only_complete_old_turns(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "core", "_tokens": 1},
            {"role": "user", "content": "old", "_tokens": 10},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1"}],
                "_tokens": 5,
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "large output",
                "_tokens": 100,
            },
            {"role": "assistant", "content": "observed", "_tokens": 5},
            {"role": "user", "content": "middle", "_tokens": 20},
            {"role": "assistant", "content": "middle answer", "_tokens": 10},
            {"role": "user", "content": "recent", "_tokens": 30},
            {"role": "assistant", "content": "recent answer", "_tokens": 20},
        ]

        preparation = prepare_compaction(
            messages,
            keep_recent_tokens=60,
            estimator=MarkerTokenEstimator(),
        )

        self.assertTrue(preparation.can_compact)
        self.assertEqual(preparation.history_start_index, 1)
        self.assertEqual(preparation.first_kept_index, 5)
        self.assertEqual(
            [message["role"] for message in preparation.protected_messages],
            ["system"],
        )
        self.assertEqual(
            [message["role"] for message in preparation.messages_to_summarize],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(
            [message["role"] for message in preparation.messages_to_keep],
            ["user", "assistant", "user", "assistant"],
        )
        self.assertEqual(preparation.estimated_history_tokens, 200)
        self.assertEqual(preparation.estimated_summarized_tokens, 120)
        self.assertEqual(preparation.estimated_kept_tokens, 80)

        messages[2]["tool_calls"][0]["id"] = "mutated"
        self.assertEqual(
            preparation.messages_to_summarize[1]["tool_calls"][0]["id"],
            "call-1",
        )

    def test_preparation_preserves_late_non_conversation_barrier(self) -> None:
        messages = [
            {"role": "system", "_tokens": 1},
            {"role": "user", "_tokens": 100},
            {"role": "assistant", "_tokens": 100},
            {"role": "system", "content": "late policy", "_tokens": 1},
            {"role": "user", "_tokens": 100},
            {"role": "assistant", "_tokens": 100},
        ]

        preparation = prepare_compaction(
            messages,
            keep_recent_tokens=10,
            estimator=MarkerTokenEstimator(),
        )

        self.assertFalse(preparation.can_compact)
        self.assertEqual(preparation.history_start_index, 4)
        self.assertEqual(len(preparation.protected_messages), 4)
        self.assertEqual(len(preparation.messages_to_keep), 2)

    async def test_agent_emits_pressure_without_mutating_or_stopping(self) -> None:
        sink = RecordingSink()
        agent = BaseAgent(
            TextModel(),
            agent_id="context-pressure",
            compaction_policy=CompactionPolicy(
                ContextBudget(
                    context_window=100,
                    reserve_tokens=20,
                    keep_recent_tokens=20,
                )
            ),
            context_token_estimator=MarkerTokenEstimator(),
            event_sink=sink,
        )
        agent.messages[0]["_tokens"] = 90

        result = await agent.run(task="pressure")

        payloads = [event.payload for event in sink.events]
        pressure = next(
            payload
            for payload in payloads
            if isinstance(payload, ContextPressureEvaluated)
        )
        self.assertTrue(pressure.decision.should_compact)
        self.assertIsNotNone(pressure.preparation)
        assert pressure.preparation is not None
        self.assertFalse(pressure.preparation.can_compact)
        self.assertEqual(
            [type(payload) for payload in payloads],
            [
                AgentStarted,
                TurnStarted,
                ContextPressureEvaluated,
                MessageCompleted,
                TurnCompleted,
                AgentFinished,
            ],
        )
        self.assertEqual(result.output, "done")
        self.assertEqual(
            [message["role"] for message in agent.messages],
            ["system", "user", "assistant"],
        )


if __name__ == "__main__":
    unittest.main()
