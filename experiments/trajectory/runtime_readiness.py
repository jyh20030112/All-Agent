#!/usr/bin/env python3
"""Execute deterministic gates for the first Runtime trajectory integration."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from phase2_evidence import run_phase2_evidence

from ejagent._trajectory import (
    CausalAction,
    CheckpointEvaluation,
    CheckpointEvaluationRequest,
    CheckpointSignal,
    CheckpointTrigger,
    EnvironmentFact,
    FactValidity,
    OnlineTrajectoryMonitor,
    ProgressStatus,
    TrajectoryContextBuffer,
    TrajectoryContextEventKind,
    TrajectoryCost,
    TrajectoryVerdict,
)
from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    CancellationToken,
    ContextRequest,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    RunIntent,
    RunLimits,
    RunSpec,
    RunStatus,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
)
from ejagent.kernel import RuntimeKernel

NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _EvaluationSpec:
    state: str
    requirement: bool | None
    constraint: bool | None = True
    evidence: tuple[str, ...] = ()
    invalidated_state: str | None = None


class _Evaluator:
    def __init__(self, specs: tuple[_EvaluationSpec, ...]) -> None:
        self._specs = list(specs)
        self.requests: list[CheckpointEvaluationRequest] = []

    async def evaluate(
        self,
        request: CheckpointEvaluationRequest,
        *,
        cancellation: CancellationToken,
    ) -> CheckpointEvaluation:
        cancellation.raise_if_cancelled()
        self.requests.append(request)
        spec = self._specs.pop(0)
        facts = [self._fact(request.checkpoint_id, spec.state)]
        if spec.invalidated_state is not None:
            facts.append(
                self._fact(
                    request.checkpoint_id,
                    spec.invalidated_state,
                    invalidated=True,
                )
            )
        return CheckpointEvaluation(
            projection_version="readiness-v1",
            state_fingerprint=f"fp:{spec.state}",
            environment_facts={"state": spec.state},
            requirements={"R": spec.requirement},
            constraints={"C": spec.constraint},
            new_evidence=spec.evidence,
            facts=tuple(facts),
            fact_capture_complete=True,
        )

    @staticmethod
    def _fact(
        checkpoint_id: str,
        state: str,
        *,
        invalidated: bool = False,
    ) -> EnvironmentFact:
        return EnvironmentFact(
            fact_id=f"{checkpoint_id}:{'old' if invalidated else 'current'}:{state}",
            subject="readiness/target",
            predicate="state",
            value=state,
            scope=("R", "C"),
            source="readiness-probe",
            observed_at=NOW,
            checkpoint_id=checkpoint_id,
            evidence_ref=f"readiness://state/{state}",
            freshness=(
                "superseded by external generation"
                if invalidated
                else "valid for this capture"
            ),
            authority="readiness target State only",
            validity=(
                FactValidity.INVALIDATED if invalidated else FactValidity.CURRENT
            ),
            invalidated_at_checkpoint=(checkpoint_id if invalidated else None),
            validity_reason=("external State changed" if invalidated else None),
        )


@dataclass(frozen=True, slots=True)
class _RuntimeReceipt:
    checkpoint_id: str
    verdict: str = "no_cycle"
    completion_allowed: bool | None = None


class _RuntimeMonitor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.signals: list[CheckpointSignal] = []
        self.closed_runs: list[str] = []

    async def capture(
        self,
        signal: CheckpointSignal,
        *,
        cancellation: CancellationToken,
    ) -> _RuntimeReceipt:
        cancellation.raise_if_cancelled()
        self.signals.append(signal)
        if self.fail:
            raise RuntimeError("readiness evaluator unavailable")
        return _RuntimeReceipt(
            checkpoint_id=f"{signal.run_id}:runtime-cp{len(self.signals) - 1}",
            completion_allowed=(
                False
                if signal.trigger is CheckpointTrigger.COMPLETION_PROPOSED
                else None
            ),
        )

    def close_run(self, run_id: str) -> object:
        self.closed_runs.append(run_id)
        return ()


class _RuntimeModel:
    def __init__(self, messages: Sequence[AssistantMessage]) -> None:
        self._messages = list(messages)

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        cancellation.raise_if_cancelled()
        yield ModelResponseCompleted(self._messages.pop(0))


class _RuntimeTools:
    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="mutate",
                description="Mutate the readiness fixture.",
                input_schema={"type": "object"},
            ),
        )

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        return ToolExecutionResult({"applied": call.id})


def _cost(
    actions: int,
    requests: int,
    tokens: int,
    elapsed_ms: int,
) -> TrajectoryCost:
    return TrajectoryCost(
        actor_actions=actions,
        model_requests=requests,
        total_tokens=tokens,
        elapsed_ms=elapsed_ms,
    )


def _baseline(run_id: str) -> CheckpointSignal:
    return CheckpointSignal(
        run_id=run_id,
        trigger=CheckpointTrigger.BASELINE,
        turn=0,
        cumulative_cost=_cost(0, 0, 0, 0),
    )


def _tool(
    run_id: str,
    turn: int,
    *,
    complete: bool = True,
) -> CheckpointSignal:
    return CheckpointSignal(
        run_id=run_id,
        trigger=CheckpointTrigger.TOOL_BATCH_COMPLETED,
        turn=turn,
        cumulative_cost=_cost(turn, turn, turn * 10, turn * 25),
        causal_actions=(CausalAction(f"call-{turn}", "retry:same"),),
        causal_batch_id=f"turn-{turn}:batch-1",
        causally_complete=complete,
        unattributed_action_ids=(() if complete else (f"unknown-{turn}",)),
        causal_exclusion_reason=(
            None if complete else "concurrent mutation attribution unavailable"
        ),
    )


async def _cycle_gates() -> dict[str, bool]:
    run_id = "cycle-run"
    evaluator = _Evaluator(tuple(_EvaluationSpec("same", False) for _ in range(3)))
    buffer = TrajectoryContextBuffer()
    monitor = OnlineTrajectoryMonitor(
        evaluator,
        update_sink=lambda update: buffer.publish(
            update.to_context_frame(
                goal="Satisfy R while preserving C.",
                next_turn=update.signal.turn + 1,
            )
        ),
    )
    cancellation = CancellationSource().token
    baseline = await monitor.capture(_baseline(run_id), cancellation=cancellation)
    suspected = await monitor.capture(_tool(run_id, 1), cancellation=cancellation)
    confirmed = await monitor.capture(_tool(run_id, 2), cancellation=cancellation)
    frame = buffer(
        ContextRequest(
            run_id=run_id,
            source_revision=0,
            turn=3,
            committed_messages=(),
        )
    )
    before = buffer(
        ContextRequest(
            run_id=run_id,
            source_revision=0,
            turn=2,
            committed_messages=(),
        )
    )
    at_boundary = buffer(
        ContextRequest(
            run_id=run_id,
            source_revision=0,
            turn=3,
            committed_messages=(),
        )
    )
    after = buffer(
        ContextRequest(
            run_id=run_id,
            source_revision=0,
            turn=4,
            committed_messages=(),
        )
    )
    return {
        "online_assessment_without_terminal_audit": (
            baseline.assessment.verdict is TrajectoryVerdict.NO_CYCLE
        ),
        "cycle_suspected_before_confirmation": (
            suspected.assessment.verdict is TrajectoryVerdict.CYCLE_SUSPECTED
        ),
        "two_cycles_confirm_non_progress": (
            confirmed.assessment.verdict is TrajectoryVerdict.NON_PROGRESS_CYCLE
            and confirmed.assessment.repeated_action_signatures == ("retry:same",)
        ),
        "confirmed_cycle_maps_to_context_event": (
            confirmed.context_event.kind is TrajectoryContextEventKind.CYCLE_CONFIRMED
        ),
        "context_is_bound_to_next_decision_only": (
            before is not None
            and before.event.kind is TrajectoryContextEventKind.CYCLE_SUSPECTED
            and at_boundary == frame
            and after is None
        ),
    }


async def _constraint_and_cost_gates() -> dict[str, bool]:
    run_id = "constraint-run"
    evaluator = _Evaluator(
        (
            _EvaluationSpec("before", False, True),
            _EvaluationSpec("after", True, False),
        )
    )
    monitor = OnlineTrajectoryMonitor(evaluator)
    cancellation = CancellationSource().token
    await monitor.capture(_baseline(run_id), cancellation=cancellation)
    update = await monitor.capture(_tool(run_id, 1), cancellation=cancellation)
    progress = update.assessment.progress[-1]
    cost = progress.cost_since_previous
    return {
        "constraint_violation_overrides_requirement_gain": (
            progress.requirement_coverage_delta == 1.0
            and progress.task_progress_delta is None
            and progress.status is ProgressStatus.REGRESSED
            and progress.newly_violated_constraints == ("C",)
        ),
        "checkpoint_cost_delta_is_complete": (
            cost.actor_actions == 1
            and cost.model_requests == 1
            and cost.total_tokens == 10
            and cost.elapsed_ms == 25
        ),
        "constraint_regression_maps_to_recovery_event": (
            update.context_event.kind is TrajectoryContextEventKind.CONSTRAINT_VIOLATED
        ),
    }


async def _boundary_gates() -> dict[str, bool]:
    run_id = "boundary-run"
    evaluator = _Evaluator(
        (
            _EvaluationSpec("s0", False),
            _EvaluationSpec("s1", False, evidence=("tool evidence",)),
            _EvaluationSpec("s1", False, evidence=("verification evidence",)),
            _EvaluationSpec("s2", False, invalidated_state="s1"),
            _EvaluationSpec("s2", True, evidence=("completion evidence",)),
        )
    )
    monitor = OnlineTrajectoryMonitor(evaluator)
    cancellation = CancellationSource().token
    signals = (
        _baseline(run_id),
        _tool(run_id, 1),
        CheckpointSignal(
            run_id,
            CheckpointTrigger.VERIFICATION_COMPLETED,
            1,
            _cost(1, 1, 10, 30),
        ),
        CheckpointSignal(
            run_id,
            CheckpointTrigger.EXTERNAL_CHANGE,
            1,
            _cost(1, 1, 10, 35),
        ),
        CheckpointSignal(
            run_id,
            CheckpointTrigger.COMPLETION_PROPOSED,
            2,
            _cost(1, 2, 20, 50),
        ),
    )
    updates = [
        await monitor.capture(signal, cancellation=cancellation) for signal in signals
    ]
    return {
        "all_semantic_checkpoint_boundaries_captured": (
            tuple(item.signal.trigger for item in updates) == tuple(CheckpointTrigger)
            and len(monitor.checkpoints(run_id)) == len(CheckpointTrigger)
        ),
        "external_change_requires_explicit_invalidation": (
            updates[3].context_event.kind
            is TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED
            and bool(updates[3].context_event.invalidated_fact_ids)
        ),
        "verified_completion_can_proceed": updates[4].completion_allowed is True,
        "run_state_has_explicit_lifecycle_close": (
            len(monitor.close_run(run_id)) == len(CheckpointTrigger)
            and not monitor.checkpoints(run_id)
        ),
    }


async def _completion_and_ambiguity_gates() -> dict[str, bool]:
    cancellation = CancellationSource().token
    completion_evaluator = _Evaluator(
        (
            _EvaluationSpec("incomplete", False),
            _EvaluationSpec("incomplete", False),
        )
    )
    completion_monitor = OnlineTrajectoryMonitor(completion_evaluator)
    await completion_monitor.capture(
        _baseline("completion-run"), cancellation=cancellation
    )
    completion = await completion_monitor.capture(
        CheckpointSignal(
            "completion-run",
            CheckpointTrigger.COMPLETION_PROPOSED,
            1,
            _cost(0, 1, 10, 20),
        ),
        cancellation=cancellation,
    )

    ambiguous_evaluator = _Evaluator(
        (_EvaluationSpec("same", False), _EvaluationSpec("same", False))
    )
    ambiguous_monitor = OnlineTrajectoryMonitor(ambiguous_evaluator)
    await ambiguous_monitor.capture(
        _baseline("ambiguous-run"), cancellation=cancellation
    )
    ambiguous = await ambiguous_monitor.capture(
        _tool("ambiguous-run", 1, complete=False),
        cancellation=cancellation,
    )
    return {
        "failed_completion_continues_same_run": (
            completion.completion_allowed is False
            and completion.context_event.kind
            is TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED
            and completion.context_event.affected_items == ("R",)
        ),
        "causal_ambiguity_fails_closed": (
            ambiguous.assessment.verdict is TrajectoryVerdict.CAUSALLY_AMBIGUOUS
        ),
    }


async def _runtime_wiring_gates() -> dict[str, bool]:
    call = ToolCall("runtime-call", "mutate", {"secret": "do-not-audit"})
    spec = RunSpec(
        run_id="runtime-readiness",
        base_revision=0,
        intent=RunIntent.TASK,
        task="verify Runtime trajectory wiring",
        messages=(SystemMessage("readiness"),),
        limits=RunLimits(max_turns=2),
        configuration_revision="trajectory-runtime-readiness-v2",
    )
    monitor = _RuntimeMonitor()
    outcome = await RuntimeKernel(
        model=_RuntimeModel(
            (AssistantMessage(tool_calls=(call,)), AssistantMessage(content="done"))
        ),
        tools=_RuntimeTools(),
        trajectory=monitor,
        clock=lambda: NOW,
        monotonic_clock=lambda: 0.0,
    ).run(spec)
    triggers = [signal.trigger for signal in monitor.signals]
    tool_signal = monitor.signals[1]
    checkpoint_records = tuple(
        record
        for record in outcome.audit_records
        if record.kind == "trajectory_checkpointed"
    )

    failing_monitor = _RuntimeMonitor(fail=True)
    fail_open = await RuntimeKernel(
        model=_RuntimeModel((AssistantMessage(content="still done"),)),
        tools=_RuntimeTools(),
        trajectory=failing_monitor,
        clock=lambda: NOW,
        monotonic_clock=lambda: 0.0,
    ).run(
        RunSpec(
            run_id="runtime-fail-open",
            base_revision=0,
            intent=RunIntent.TASK,
            task="verify fail open",
            messages=(SystemMessage("readiness"),),
            configuration_revision="trajectory-runtime-readiness-v2",
        )
    )
    default_outcome = await RuntimeKernel(
        model=_RuntimeModel((AssistantMessage(content="default"),)),
        tools=_RuntimeTools(),
        clock=lambda: NOW,
    ).run(
        RunSpec(
            run_id="runtime-default",
            base_revision=0,
            intent=RunIntent.TASK,
            task="verify default path",
            messages=(SystemMessage("readiness"),),
            configuration_revision="trajectory-runtime-readiness-v2",
        )
    )
    return {
        "runtime_default_path_has_no_trajectory_events": not any(
            record.kind.startswith("trajectory_")
            for record in default_outcome.audit_records
        ),
        "runtime_captures_declared_boundaries_in_order": triggers
        == [
            CheckpointTrigger.BASELINE,
            CheckpointTrigger.TOOL_BATCH_COMPLETED,
            CheckpointTrigger.COMPLETION_PROPOSED,
        ],
        "runtime_tool_batch_has_complete_redacted_causality": (
            tool_signal.causally_complete
            and tool_signal.causal_batch_id == "turn-1:tool-batch"
            and tool_signal.cumulative_cost.actor_actions == 1
            and tool_signal.cumulative_cost.model_requests == 1
            and len(tool_signal.causal_actions) == 1
            and "do-not-audit" not in tool_signal.causal_actions[0].signature
        ),
        "runtime_completion_advice_is_observation_only": (
            outcome.result.status is RunStatus.COMPLETED
            and len(checkpoint_records) == 3
            and checkpoint_records[-1].payload["completion_allowed"] is False
        ),
        "runtime_monitor_failure_is_fail_open": (
            fail_open.result.status is RunStatus.COMPLETED
            and [signal.trigger for signal in failing_monitor.signals]
            == [CheckpointTrigger.BASELINE]
            and any(
                record.kind == "trajectory_capture_failed"
                for record in fail_open.audit_records
            )
        ),
        "runtime_closes_monitor_on_exit": (
            monitor.closed_runs == ["runtime-readiness"]
            and failing_monitor.closed_runs == ["runtime-fail-open"]
        ),
    }


def _dependency_gates(root: Path) -> dict[str, bool]:
    runtime_source = (root / "src/ejagent/kernel/runtime.py").read_text(
        encoding="utf-8"
    )
    stable_sources = tuple(
        path
        for package in ("contracts", "context", "kernel", "harness")
        for path in (root / "src/ejagent" / package).rglob("*.py")
    )
    return {
        "runtime_uses_stable_trajectory_seam": (
            "from ejagent.kernel.trajectory import" in runtime_source
        ),
        "stable_runtime_layers_do_not_depend_on_internal_trajectory": all(
            "ejagent._trajectory" not in path.read_text(encoding="utf-8")
            for path in stable_sources
        ),
    }


async def run_runtime_readiness(root: Path) -> dict[str, object]:
    gates: dict[str, bool] = {}
    gates.update(await _cycle_gates())
    gates.update(await _constraint_and_cost_gates())
    gates.update(await _boundary_gates())
    gates.update(await _completion_and_ambiguity_gates())
    gates.update(await _runtime_wiring_gates())
    gates.update(_dependency_gates(root))
    phase2 = run_phase2_evidence()
    gates["phase2_evidence_remains_green"] = bool(phase2["all_gates_passed"])
    return {
        "evaluation": "trajectory-runtime-integration-v2",
        "runtime_modified": True,
        "checkpoint_triggers": [item.value for item in CheckpointTrigger],
        "integration_mode": "Runtime opt-in observation and context projection",
        "enforcement_enabled": False,
        "phase2_evidence": phase2["experiment"],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = asyncio.run(run_runtime_readiness(root))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
