from __future__ import annotations

import unittest
from datetime import UTC, datetime

from ejagent._trajectory import (
    CausalAction,
    CheckpointEvaluation,
    CheckpointEvaluationRequest,
    CheckpointProtocolError,
    CheckpointSignal,
    CheckpointTrigger,
    EnvironmentFact,
    FactValidity,
    OnlineTrajectoryMonitor,
    TrajectoryContextBuffer,
    TrajectoryContextEventKind,
    TrajectoryCost,
    TrajectoryVerdict,
)
from ejagent.contracts import (
    CancellationSource,
    CancellationToken,
    ContextRequest,
    SystemMessage,
)

NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)


class _Evaluator:
    def __init__(self, states: list[tuple[str, bool, tuple[str, ...]]]) -> None:
        self._states = states
        self.requests: list[CheckpointEvaluationRequest] = []

    async def evaluate(
        self,
        request: CheckpointEvaluationRequest,
        *,
        cancellation: CancellationToken,
    ) -> CheckpointEvaluation:
        cancellation.raise_if_cancelled()
        self.requests.append(request)
        state, requirement, evidence = self._states.pop(0)
        fact = EnvironmentFact(
            fact_id=f"{request.checkpoint_id}:state",
            subject="fixture/target",
            predicate="state",
            value=state,
            scope=("R1",),
            source="fixture-verifier",
            observed_at=NOW,
            checkpoint_id=request.checkpoint_id,
            evidence_ref=f"fixture://state/{state}",
            freshness="valid for this capture",
            authority="fixture target State only",
        )
        facts = (fact,)
        if request.signal.trigger is CheckpointTrigger.EXTERNAL_CHANGE:
            facts = (
                fact,
                EnvironmentFact(
                    fact_id=f"{request.checkpoint_id}:invalidated-prior-state",
                    subject="fixture/target",
                    predicate="state",
                    value="s1",
                    scope=("R1",),
                    source="fixture-verifier",
                    observed_at=NOW,
                    checkpoint_id=request.checkpoint_id,
                    evidence_ref="fixture://state/s1",
                    freshness="superseded by external generation",
                    authority="fixture target State only",
                    validity=FactValidity.INVALIDATED,
                    invalidated_at_checkpoint=request.checkpoint_id,
                    validity_reason="external State changed",
                ),
            )
        return CheckpointEvaluation(
            projection_version="fixture-v1",
            state_fingerprint=state,
            environment_facts={"state": state},
            requirements={"R1": requirement},
            constraints={"C1": True},
            new_evidence=evidence,
            facts=facts,
            fact_capture_complete=True,
        )


def _cost(
    actions: int,
    *,
    requests: int = 0,
    tokens: int = 0,
    elapsed_ms: int = 0,
) -> TrajectoryCost:
    return TrajectoryCost(
        actor_actions=actions,
        model_requests=requests,
        total_tokens=tokens,
        elapsed_ms=elapsed_ms,
    )


def _baseline(run_id: str = "online-run") -> CheckpointSignal:
    return CheckpointSignal(
        run_id=run_id,
        trigger=CheckpointTrigger.BASELINE,
        turn=0,
        cumulative_cost=_cost(0),
    )


def _tool_signal(sequence: int, *, complete: bool = True) -> CheckpointSignal:
    return CheckpointSignal(
        run_id="online-run",
        trigger=CheckpointTrigger.TOOL_BATCH_COMPLETED,
        turn=sequence,
        cumulative_cost=_cost(
            sequence,
            requests=sequence,
            tokens=sequence * 10,
            elapsed_ms=sequence * 25,
        ),
        causal_actions=(CausalAction(f"call-{sequence}", "retry:same"),),
        causal_batch_id=f"turn-{sequence}:batch-1",
        causally_complete=complete,
        unattributed_action_ids=(() if complete else (f"call-{sequence}:unknown",)),
        causal_exclusion_reason=(
            None if complete else "concurrent mutation attribution unavailable"
        ),
    )


class OnlineTrajectoryMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_assesses_suspicion_then_confirmation_before_run_terminal(
        self,
    ) -> None:
        evaluator = _Evaluator(
            [
                ("same", False, ()),
                ("same", False, ()),
                ("same", False, ()),
            ]
        )
        buffer = TrajectoryContextBuffer()
        closed_histories: list[tuple[str, tuple[object, ...]]] = []

        def close_sink(run_id: str, checkpoints: tuple[object, ...]) -> None:
            closed_histories.append((run_id, checkpoints))
            buffer.close_run(run_id)

        monitor = OnlineTrajectoryMonitor(
            evaluator,
            update_sink=lambda update: buffer.publish(
                update.to_context_frame(
                    goal="Make R1 true without violating C1.",
                    next_turn=update.signal.turn + 1,
                )
            ),
            run_close_sink=close_sink,
        )
        cancellation = CancellationSource().token

        baseline = await monitor.capture(_baseline(), cancellation=cancellation)
        suspected = await monitor.capture(_tool_signal(1), cancellation=cancellation)
        confirmed = await monitor.capture(_tool_signal(2), cancellation=cancellation)

        self.assertEqual(baseline.assessment.verdict, TrajectoryVerdict.NO_CYCLE)
        self.assertEqual(
            suspected.assessment.verdict, TrajectoryVerdict.CYCLE_SUSPECTED
        )
        self.assertEqual(
            confirmed.assessment.verdict,
            TrajectoryVerdict.NON_PROGRESS_CYCLE,
        )
        self.assertEqual(confirmed.checkpoint.capture_trigger, "tool_batch_completed")
        self.assertEqual(confirmed.checkpoint.turn, 2)
        frame = buffer(
            ContextRequest(
                run_id="online-run",
                source_revision=0,
                turn=3,
                committed_messages=(SystemMessage("stable"),),
            )
        )
        self.assertIsNotNone(frame)

        def request(turn: int) -> ContextRequest:
            return ContextRequest(
                run_id="online-run",
                source_revision=0,
                turn=turn,
                committed_messages=(SystemMessage("stable"),),
            )

        before = buffer(request(2))
        self.assertIsNotNone(before)
        assert before is not None
        self.assertEqual(
            before.event.kind,
            TrajectoryContextEventKind.CYCLE_SUSPECTED,
        )
        self.assertNotEqual(before, frame)
        self.assertEqual(buffer(request(3)), frame)
        self.assertIsNone(buffer(request(4)))
        self.assertEqual(len(monitor.checkpoints("online-run")), 3)
        closed = monitor.close_run("online-run")
        self.assertEqual(len(closed), 3)
        self.assertEqual(monitor.checkpoints("online-run"), ())
        self.assertIsNone(buffer(request(3)))
        self.assertEqual(closed_histories, [("online-run", closed)])

    async def test_captures_every_declared_semantic_boundary_in_order(self) -> None:
        evaluator = _Evaluator(
            [
                ("s0", False, ()),
                ("s1", False, ("tool observation",)),
                ("s1", False, ("verifier result",)),
                ("s2", False, ("external change",)),
                ("s2", True, ("completion audit",)),
            ]
        )
        monitor = OnlineTrajectoryMonitor(evaluator)
        cancellation = CancellationSource().token
        signals = (
            _baseline(),
            _tool_signal(1),
            CheckpointSignal(
                "online-run",
                CheckpointTrigger.VERIFICATION_COMPLETED,
                1,
                _cost(1, requests=1, tokens=10, elapsed_ms=30),
            ),
            CheckpointSignal(
                "online-run",
                CheckpointTrigger.EXTERNAL_CHANGE,
                1,
                _cost(1, requests=1, tokens=10, elapsed_ms=35),
            ),
            CheckpointSignal(
                "online-run",
                CheckpointTrigger.COMPLETION_PROPOSED,
                2,
                _cost(1, requests=2, tokens=20, elapsed_ms=50),
            ),
        )

        updates = [
            await monitor.capture(signal, cancellation=cancellation)
            for signal in signals
        ]

        self.assertEqual(
            [item.checkpoint.capture_trigger for item in updates],
            [item.value for item in CheckpointTrigger],
        )
        self.assertEqual(
            [item.checkpoint.checkpoint_id for item in updates],
            [f"online-run:cp{index}" for index in range(5)],
        )
        self.assertEqual(
            updates[-1].assessment.progress[-1].cost_since_previous.model_requests,
            1,
        )
        self.assertTrue(updates[-1].completion_allowed)

    async def test_failed_completion_audit_produces_continue_event(self) -> None:
        evaluator = _Evaluator([("incomplete", False, ()), ("incomplete", False, ())])
        monitor = OnlineTrajectoryMonitor(evaluator)
        cancellation = CancellationSource().token
        await monitor.capture(_baseline(), cancellation=cancellation)

        update = await monitor.capture(
            CheckpointSignal(
                "online-run",
                CheckpointTrigger.COMPLETION_PROPOSED,
                1,
                _cost(0, requests=1, tokens=10, elapsed_ms=20),
            ),
            cancellation=cancellation,
        )

        self.assertFalse(update.completion_allowed)
        self.assertEqual(
            update.context_event.kind,
            TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED,
        )
        self.assertEqual(update.context_event.affected_items, ("R1",))

    async def test_excludes_causally_ambiguous_batch_from_cycle_proof(self) -> None:
        evaluator = _Evaluator([("same", False, ()), ("same", False, ())])
        monitor = OnlineTrajectoryMonitor(evaluator)
        cancellation = CancellationSource().token
        await monitor.capture(_baseline(), cancellation=cancellation)

        update = await monitor.capture(
            _tool_signal(1, complete=False),
            cancellation=cancellation,
        )

        self.assertEqual(
            update.assessment.verdict,
            TrajectoryVerdict.CAUSALLY_AMBIGUOUS,
        )
        self.assertEqual(
            update.assessment.candidate_checkpoint_ids,
            ("online-run:cp1",),
        )

    async def test_rejects_missing_baseline_and_lost_known_cost(self) -> None:
        evaluator = _Evaluator([("same", False, ()), ("same", False, ())])
        monitor = OnlineTrajectoryMonitor(evaluator)
        cancellation = CancellationSource().token

        with self.assertRaisesRegex(CheckpointProtocolError, "first.*baseline"):
            await monitor.capture(_tool_signal(1), cancellation=cancellation)

        await monitor.capture(_baseline(), cancellation=cancellation)
        regressed_cost = CheckpointSignal(
            "online-run",
            CheckpointTrigger.COMPLETION_PROPOSED,
            1,
            _cost(0, requests=0, tokens=0, elapsed_ms=0),
        )
        await monitor.capture(regressed_cost, cancellation=cancellation)
        losing_known_cost = CheckpointSignal(
            "online-run",
            CheckpointTrigger.COMPLETION_PROPOSED,
            2,
            TrajectoryCost(actor_actions=0, model_requests=0),
        )
        with self.assertRaisesRegex(CheckpointProtocolError, "became unavailable"):
            await monitor.capture(losing_known_cost, cancellation=cancellation)


if __name__ == "__main__":
    unittest.main()
