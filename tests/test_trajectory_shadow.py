from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from ejagent import AgentHarness, FunctionToolExecutor
from ejagent._trajectory import (
    EnvironmentFact,
    FactValidity,
    ProgressStatus,
    ShadowTrajectoryAnalyzer,
    ShadowTrajectoryObserver,
    TrajectoryCheckpoint,
    TrajectoryCost,
    TrajectoryVerdict,
)
from ejagent.contracts import (
    AssistantMessage,
    AuditRecord,
    CancellationToken,
    ModelRequest,
    ModelResponseCompleted,
    ModelUsage,
    RunAudit,
    RunResult,
    RunStatus,
    StopReason,
)

NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)


class _TextModel:
    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelResponseCompleted]:
        del request
        cancellation.raise_if_cancelled()
        yield ModelResponseCompleted(
            AssistantMessage(content="done"),
            ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def _audit(*records: AuditRecord) -> RunAudit:
    return RunAudit(
        result=RunResult(
            run_id="run-1",
            status=RunStatus.COMPLETED,
            stop_reason=StopReason.TEXT_RESPONSE,
            turns=1,
            output="done",
        ),
        base_revision=0,
        resulting_revision=1,
        committed=True,
        records=records,
    )


def _record(sequence: int, kind: str, payload: dict[str, object]) -> AuditRecord:
    return AuditRecord(
        run_id="run-1",
        sequence=sequence,
        kind=kind,
        occurred_at=NOW + timedelta(seconds=sequence),
        payload=payload,
    )


def _checkpoint(
    sequence: int,
    state: str,
    *,
    requirements: dict[str, bool | None],
    action: str | None = None,
    evidence: tuple[str, ...] = (),
    facts: dict[str, object] | None = None,
    action_count: int | None = None,
    causally_complete: bool = True,
    fact_records: tuple[EnvironmentFact, ...] = (),
    fact_capture_complete: bool = False,
    causal_batch_id: str | None = None,
    action_signatures: tuple[str, ...] | None = None,
) -> TrajectoryCheckpoint:
    return TrajectoryCheckpoint(
        checkpoint_id=f"cp{sequence}",
        projection_version="test-v1",
        state_fingerprint=state,
        environment_facts=facts or {"state": state},
        requirements=requirements,
        constraints={"C1": True},
        new_evidence=evidence,
        actor_action_count=sequence if action_count is None else action_count,
        causal_action_signatures=(
            (() if action is None else (action,))
            if action_signatures is None
            else action_signatures
        ),
        causally_complete=causally_complete,
        facts=fact_records,
        fact_capture_complete=fact_capture_complete,
        causal_batch_id=causal_batch_id,
        unattributed_action_ids=(() if causally_complete else ("call-unknown",)),
        causal_exclusion_reason=(
            None if causally_complete else "concurrent mutation attribution unavailable"
        ),
    )


def _environment_fact(
    fact_id: str,
    value: str,
    *,
    validity: FactValidity = FactValidity.CURRENT,
) -> EnvironmentFact:
    return EnvironmentFact(
        fact_id=fact_id,
        subject="service/config",
        predicate="mode",
        value=value,
        scope=("R1",),
        source="config-reader",
        observed_at=NOW,
        checkpoint_id="cp0",
        evidence_ref="config://mode",
        freshness="valid until config generation changes",
        authority="service mode only",
        validity=validity,
        validity_reason=(
            "freshness window elapsed" if validity is not FactValidity.CURRENT else None
        ),
    )


class ShadowTrajectoryAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = ShadowTrajectoryAnalyzer(max_period=3)

    def test_requires_two_complete_action_and_state_cycles_for_confirmation(
        self,
    ) -> None:
        checkpoints = (
            _checkpoint(0, "S0", requirements={"R1": False, "R2": True}),
            _checkpoint(
                1,
                "S1",
                requirements={"R1": True, "R2": False},
                action="A",
                evidence=("global strict policy breaks refresh",),
            ),
            _checkpoint(
                2,
                "S0",
                requirements={"R1": False, "R2": True},
                action="B",
            ),
            _checkpoint(
                3,
                "S1",
                requirements={"R1": True, "R2": False},
                action="A",
            ),
            _checkpoint(
                4,
                "S0",
                requirements={"R1": False, "R2": True},
                action="B",
            ),
        )

        suspected = self.analyzer.analyze(_audit(), checkpoints[:4])
        confirmed = self.analyzer.analyze(_audit(), checkpoints)

        self.assertEqual(suspected.verdict, TrajectoryVerdict.CYCLE_SUSPECTED)
        self.assertEqual(
            suspected.candidate_checkpoint_ids, ("cp0", "cp1", "cp2", "cp3")
        )
        self.assertFalse(suspected.repeated_action_path)
        self.assertEqual(confirmed.verdict, TrajectoryVerdict.NON_PROGRESS_CYCLE)
        self.assertEqual(confirmed.period, 2)
        self.assertEqual(
            confirmed.candidate_checkpoint_ids,
            ("cp0", "cp1", "cp2", "cp3", "cp4"),
        )
        self.assertTrue(confirmed.repeated_action_path)
        self.assertEqual(confirmed.task_progress_over_repeated_window, 0.0)
        self.assertEqual(confirmed.progress[-1].regressed_requirements, ("R1",))

    def test_online_assessment_does_not_require_a_terminal_run_audit(self) -> None:
        checkpoints = (
            _checkpoint(0, "same", requirements={"R1": False}),
            _checkpoint(1, "same", requirements={"R1": False}, action="retry"),
            _checkpoint(2, "same", requirements={"R1": False}, action="retry"),
        )

        assessment = self.analyzer.assess(checkpoints)
        report = self.analyzer.analyze(_audit(), checkpoints)

        self.assertEqual(assessment.verdict, TrajectoryVerdict.NON_PROGRESS_CYCLE)
        self.assertEqual(assessment.verdict, report.verdict)
        self.assertEqual(assessment.progress, report.progress)
        self.assertEqual(
            assessment.candidate_checkpoint_ids,
            report.candidate_checkpoint_ids,
        )

    def test_requirement_gain_is_not_task_progress_when_constraint_regresses(
        self,
    ) -> None:
        baseline = TrajectoryCheckpoint(
            checkpoint_id="cp0",
            projection_version="test-v1",
            state_fingerprint="S0",
            environment_facts={"state": "S0"},
            requirements={"R1": False},
            constraints={"C1": True},
            cumulative_cost=TrajectoryCost(
                actor_actions=0,
                model_requests=0,
                total_tokens=0,
                elapsed_ms=0,
            ),
        )
        regressed = TrajectoryCheckpoint(
            checkpoint_id="cp1",
            projection_version="test-v1",
            state_fingerprint="S1",
            environment_facts={"state": "S1"},
            requirements={"R1": True},
            constraints={"C1": False},
            actor_action_count=1,
            causal_action_signatures=("edit",),
            cumulative_cost=TrajectoryCost(
                actor_actions=1,
                model_requests=1,
                total_tokens=12,
                elapsed_ms=25,
            ),
        )

        progress = self.analyzer.assess((baseline, regressed)).progress[-1]

        self.assertEqual(progress.requirement_coverage_delta, 1.0)
        self.assertIsNone(progress.task_progress_delta)
        self.assertEqual(progress.status, ProgressStatus.REGRESSED)
        self.assertEqual(progress.newly_violated_constraints, ("C1",))
        self.assertEqual(progress.violated_constraints, ("C1",))
        self.assertEqual(progress.cost_since_previous.actor_actions, 1)
        self.assertEqual(progress.cost_since_previous.model_requests, 1)
        self.assertEqual(progress.cost_since_previous.total_tokens, 12)
        self.assertEqual(progress.cost_since_previous.elapsed_ms, 25)

    def test_fingerprint_match_without_equal_facts_cannot_prove_recurrence(
        self,
    ) -> None:
        checkpoints = (
            _checkpoint(0, "same", requirements={"R1": False}, facts={"value": 0}),
            _checkpoint(
                1,
                "same",
                requirements={"R1": False},
                facts={"value": 1},
                action="poll",
            ),
            _checkpoint(
                2,
                "same",
                requirements={"R1": False},
                facts={"value": 2},
                action="poll",
            ),
        )

        report = self.analyzer.analyze(_audit(), checkpoints)

        self.assertEqual(report.verdict, TrajectoryVerdict.NO_CYCLE)

    def test_incomplete_checkpoint_cannot_confirm_cycle(self) -> None:
        checkpoints = (
            _checkpoint(0, "same", requirements={"R1": False}),
            _checkpoint(1, "same", requirements={"R1": False}, action="retry"),
            _checkpoint(
                2,
                "same",
                requirements={"R1": False},
                action="retry",
                causally_complete=False,
            ),
        )

        report = self.analyzer.analyze(_audit(), checkpoints)

        self.assertEqual(report.verdict, TrajectoryVerdict.CAUSALLY_AMBIGUOUS)
        self.assertEqual(report.candidate_checkpoint_ids, ("cp2",))

    def test_explicit_facts_must_be_complete_and_current(self) -> None:
        stale_fact = _environment_fact(
            "mode-stale",
            "strict",
            validity=FactValidity.STALE,
        )
        checkpoints = (
            _checkpoint(
                0,
                "same",
                requirements={"R1": False},
                fact_records=(stale_fact,),
                fact_capture_complete=True,
            ),
            _checkpoint(
                1,
                "same",
                requirements={"R1": False},
                action="retry",
                fact_records=(stale_fact,),
                fact_capture_complete=True,
            ),
            _checkpoint(
                2,
                "same",
                requirements={"R1": False},
                action="retry",
                fact_records=(stale_fact,),
                fact_capture_complete=True,
            ),
        )

        report = self.analyzer.analyze(_audit(), checkpoints)

        self.assertEqual(report.verdict, TrajectoryVerdict.INSUFFICIENT_EVIDENCE)
        self.assertTrue(any("validity is stale" in item for item in report.diagnostics))

    def test_complete_concurrent_batches_can_confirm_recurrence(self) -> None:
        current_fact = _environment_fact("mode-current", "strict")
        checkpoints = (
            _checkpoint(
                0,
                "same",
                requirements={"R1": False},
                fact_records=(current_fact,),
                fact_capture_complete=True,
            ),
            _checkpoint(
                1,
                "same",
                requirements={"R1": False},
                fact_records=(current_fact,),
                fact_capture_complete=True,
                causal_batch_id="batch-1",
                action_signatures=("write:primary", "write:replica"),
            ),
            _checkpoint(
                2,
                "same",
                requirements={"R1": False},
                fact_records=(current_fact,),
                fact_capture_complete=True,
                causal_batch_id="batch-2",
                action_signatures=("write:primary", "write:replica"),
            ),
        )

        report = self.analyzer.analyze(_audit(), checkpoints)

        self.assertEqual(report.verdict, TrajectoryVerdict.NON_PROGRESS_CYCLE)
        self.assertTrue(report.repeated_action_path)

    def test_allows_a_goal_without_constraints(self) -> None:
        checkpoint = TrajectoryCheckpoint(
            checkpoint_id="cp0",
            projection_version="test-v1",
            state_fingerprint="S0",
            environment_facts={"state": "S0"},
            requirements={"R1": False},
            constraints={},
        )

        report = self.analyzer.analyze(_audit(), (checkpoint,))

        self.assertEqual(report.verdict, TrajectoryVerdict.NO_CYCLE)
        self.assertEqual(dict(report.progress[0].constraints), {})

    def test_preserves_required_healthy_controls(self) -> None:
        scenarios = {
            "productive polling": (
                _checkpoint(0, "10%", requirements={"done": False}),
                _checkpoint(
                    1,
                    "60%",
                    requirements={"done": False},
                    action="poll",
                    evidence=("60%",),
                ),
                _checkpoint(
                    2,
                    "done",
                    requirements={"done": True},
                    action="poll",
                    evidence=("complete",),
                ),
            ),
            "productive edit verify": (
                _checkpoint(0, "fail-2", requirements={"R1": False, "R2": False}),
                _checkpoint(
                    1,
                    "fail-1",
                    requirements={"R1": True, "R2": False},
                    action="edit-verify",
                ),
                _checkpoint(
                    2,
                    "pass",
                    requirements={"R1": True, "R2": True},
                    action="edit-verify",
                ),
            ),
            "evidence gaining exploration": (
                _checkpoint(0, "world", requirements={"R1": False}),
                _checkpoint(
                    1,
                    "world",
                    requirements={"R1": False},
                    action="inspect",
                    evidence=("A excluded",),
                ),
                _checkpoint(
                    2,
                    "world",
                    requirements={"R1": False},
                    action="inspect",
                    evidence=("B narrowed",),
                ),
            ),
            "legitimate retry": (
                _checkpoint(0, "down", requirements={"R1": False}),
                _checkpoint(
                    1,
                    "ready",
                    requirements={"R1": True},
                    action="retry",
                    evidence=("external recovery",),
                ),
            ),
        }

        for name, checkpoints in scenarios.items():
            with self.subTest(name=name):
                report = self.analyzer.analyze(_audit(), checkpoints)
                self.assertEqual(report.verdict, TrajectoryVerdict.NO_CYCLE)

    def test_normalizes_proposal_order_and_actual_completion_order(self) -> None:
        audit = _audit(
            _record(
                1,
                "assistant_message",
                {
                    "turn": 1,
                    "content": None,
                    "tool_calls": [
                        {"id": "call-a", "name": "read", "arguments": {"path": "a"}},
                        {"id": "call-b", "name": "read", "arguments": {"path": "b"}},
                    ],
                },
            ),
            _record(
                2,
                "tool_started",
                {
                    "turn": 1,
                    "tool_call_id": "call-a",
                    "tool_name": "read",
                    "arguments": {"path": "a"},
                },
            ),
            _record(
                3,
                "tool_started",
                {
                    "turn": 1,
                    "tool_call_id": "call-b",
                    "tool_name": "read",
                    "arguments": {"path": "b"},
                },
            ),
            _record(
                4,
                "tool_completed",
                {
                    "turn": 1,
                    "tool_call_id": "call-b",
                    "tool_name": "read",
                    "control": "continue",
                    "is_error": False,
                    "result": {"value": "b"},
                },
            ),
            _record(
                5,
                "tool_completed",
                {
                    "turn": 1,
                    "tool_call_id": "call-a",
                    "tool_name": "read",
                    "control": "continue",
                    "is_error": False,
                    "result": {"value": "a"},
                },
            ),
        )

        report = self.analyzer.analyze(audit, ())

        self.assertEqual(report.verdict, TrajectoryVerdict.INSUFFICIENT_EVIDENCE)
        self.assertEqual(
            [item.tool_call_id for item in report.actions], ["call-a", "call-b"]
        )
        self.assertEqual(
            [item.tool_call_id for item in report.observations],
            ["call-b", "call-a"],
        )
        self.assertEqual(report.actions[0].started_audit_sequence, 2)
        self.assertEqual(report.actions[0].completed_audit_sequence, 5)


class ShadowTrajectoryObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_observer_joins_post_commit_audit_with_host_facts(
        self,
    ) -> None:
        checkpoints = (
            _checkpoint(0, "S0", requirements={"R1": False}),
            _checkpoint(1, "S1", requirements={"R1": True}, action="edit"),
        )
        reports = []

        async def sink(report: object) -> None:
            reports.append(report)

        observer = ShadowTrajectoryObserver(
            checkpoint_source=lambda run_id: checkpoints if run_id == "run-1" else (),
            report_sink=sink,
        )

        harness = AgentHarness(
            agent_id="shadow-agent",
            model=_TextModel(),
            tools=FunctionToolExecutor(),
            observers=(observer,),
            run_id_factory=lambda: "run-1",
        )

        async with harness:
            outcome = await harness.run("finish")

        self.assertTrue(outcome.result.succeeded)
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].committed)
        self.assertEqual(reports[0].base_revision, 0)
        self.assertEqual(reports[0].resulting_revision, 1)
        self.assertEqual(reports[0].total_tokens, 2)
        self.assertEqual(reports[0].verdict, TrajectoryVerdict.NO_CYCLE)


if __name__ == "__main__":
    unittest.main()
