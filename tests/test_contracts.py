from __future__ import annotations

import unittest
from datetime import UTC, datetime

from ejagent import RunStatus as LegacyRunStatus
from ejagent import RunUsage as LegacyRunUsage
from ejagent import ToolControl as LegacyToolControl
from ejagent import ToolEffect as LegacyToolEffect
from ejagent.contracts import (
    AssistantMessage,
    AuditRecord,
    ContextSummary,
    FailureCode,
    RunDelta,
    RunFailure,
    RunIntent,
    RunLimits,
    RunOutcome,
    RunPhase,
    RunResult,
    RunSpec,
    RunStatus,
    RunUsage,
    StopReason,
    ToolCall,
    ToolControl,
    ToolEffect,
    ToolResultMessage,
    UserMessage,
    thaw_json_value,
)


class MessageContractTests(unittest.TestCase):
    def test_tool_call_recursively_freezes_detached_arguments(self) -> None:
        source = {"path": "README.md", "options": {"lines": [1, 2]}}

        call = ToolCall(id="call-1", name="read", arguments=source)
        source["path"] = "changed"
        options = source["options"]
        assert isinstance(options, dict)
        lines = options["lines"]
        assert isinstance(lines, list)
        lines.append(3)

        self.assertEqual(call.arguments["path"], "README.md")
        self.assertEqual(
            thaw_json_value(call.arguments),
            {"path": "README.md", "options": {"lines": [1, 2]}},
        )
        with self.assertRaises(TypeError):
            call.arguments["path"] = "forbidden"  # type: ignore[index]

    def test_assistant_requires_text_or_tool_calls(self) -> None:
        with self.assertRaisesRegex(ValueError, "text or tool calls"):
            AssistantMessage()

        message = AssistantMessage(tool_calls=(ToolCall(id="call-1", name="lookup"),))

        self.assertEqual(message.tool_calls[0].name, "lookup")

    def test_tool_result_is_typed_and_immutable(self) -> None:
        source = {"ok": True, "items": ["a"]}

        message = ToolResultMessage(
            tool_call_id="call-1",
            tool_name="lookup",
            result=source,
        )
        source["ok"] = False

        self.assertEqual(
            thaw_json_value(message.result),
            {"ok": True, "items": ["a"]},
        )

    def test_context_summary_is_not_a_conversation_message(self) -> None:
        summary = ContextSummary(
            source_revision_start=1,
            source_revision_end=4,
            content="Earlier work was summarized.",
            compactor_id="test",
        )

        self.assertEqual(summary.source_revision_end, 4)


class RunContractTests(unittest.TestCase):
    def test_run_spec_freezes_inputs_at_one_revision(self) -> None:
        messages = [UserMessage("existing input")]
        metadata = {"trace": {"labels": ["contract"]}}

        spec = RunSpec(
            run_id="run-1",
            base_revision=7,
            intent=RunIntent.TASK,
            task="new task",
            messages=messages,
            configuration_revision="config-3",
            metadata=metadata,
        )
        messages.append(UserMessage("late mutation"))
        trace = metadata["trace"]
        assert isinstance(trace, dict)
        labels = trace["labels"]
        assert isinstance(labels, list)
        labels.append("late")

        self.assertEqual(spec.messages, (UserMessage("existing input"),))
        self.assertEqual(
            thaw_json_value(spec.metadata),
            {"trace": {"labels": ["contract"]}},
        )

    def test_continue_spec_rejects_a_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain a task"):
            RunSpec(
                run_id="run-1",
                base_revision=0,
                intent=RunIntent.CONTINUE,
                task="unexpected",
                messages=(),
            )

    def test_limits_reject_boolean_in_integer_fields(self) -> None:
        with self.assertRaisesRegex(TypeError, "max_turns must be an integer"):
            RunLimits(max_turns=True)  # type: ignore[arg-type]

    def test_delta_proposes_the_next_revision(self) -> None:
        delta = RunDelta(
            base_revision=4,
            messages=(UserMessage("task"),),
        )

        self.assertEqual(delta.next_revision, 5)

    def test_failed_outcome_requires_structured_failure(self) -> None:
        result = RunResult(
            run_id="run-1",
            status=RunStatus.FAILED,
            stop_reason=StopReason.RUNTIME_ERROR,
            turns=1,
        )

        with self.assertRaisesRegex(ValueError, "requires a RunFailure"):
            RunOutcome(result=result, delta=RunDelta(base_revision=0))

        outcome = RunOutcome(
            result=result,
            delta=RunDelta(base_revision=0),
            failure=RunFailure(
                phase=RunPhase.RUNTIME,
                code=FailureCode.RUNTIME_ERROR,
                message="expected failure",
            ),
        )

        self.assertEqual(outcome.failure.code, FailureCode.RUNTIME_ERROR)

    def test_audit_records_are_ordered_and_belong_to_the_run(self) -> None:
        result = RunResult(
            run_id="run-1",
            status=RunStatus.COMPLETED,
            stop_reason=StopReason.TEXT_RESPONSE,
            turns=1,
            output="done",
        )
        later = AuditRecord(
            run_id="run-1",
            sequence=2,
            kind="run_finished",
            occurred_at=datetime.now(UTC),
        )
        earlier = AuditRecord(
            run_id="run-1",
            sequence=1,
            kind="run_started",
            occurred_at=datetime.now(UTC),
        )

        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            RunOutcome(
                result=result,
                delta=RunDelta(base_revision=0),
                audit_records=(later, earlier),
            )

    def test_legacy_paths_reexport_canonical_status_and_usage(self) -> None:
        self.assertIs(LegacyRunStatus, RunStatus)
        self.assertIs(LegacyRunUsage, RunUsage)
        self.assertIs(LegacyToolControl, ToolControl)
        self.assertIs(LegacyToolEffect, ToolEffect)


if __name__ == "__main__":
    unittest.main()
