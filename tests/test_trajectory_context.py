from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from ejagent._trajectory import (
    EnvironmentFact,
    FactValidity,
    ProgressSnapshot,
    TrajectoryCheckpoint,
    TrajectoryContextEvent,
    TrajectoryContextEventKind,
    TrajectoryContextFrame,
    TrajectoryContextPipeline,
)
from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    CancellationToken,
    ContextProtocolError,
    ContextRequest,
    ContextView,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    RunIntent,
    RunLimits,
    RunSpec,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutor,
    TransientInstruction,
    UserMessage,
)
from ejagent.contracts.json import JsonValue
from ejagent.kernel import RuntimeKernel

NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)


def _fact(
    fact_id: str,
    value: JsonValue,
    *,
    validity: FactValidity = FactValidity.CURRENT,
) -> EnvironmentFact:
    return EnvironmentFact(
        fact_id=fact_id,
        subject="deployment/router",
        predicate="active_color",
        value=value,
        scope=("R-route",),
        source="deployment-api",
        observed_at=NOW,
        checkpoint_id="cp2" if validity is FactValidity.CURRENT else "cp1",
        evidence_ref=f"deployment://evidence/{fact_id}",
        freshness="valid until the next deployment generation",
        authority="active routing color only",
        validity=validity,
        invalidated_at_checkpoint=(
            "cp2" if validity is FactValidity.INVALIDATED else None
        ),
        validity_reason=(
            "deployment generation changed"
            if validity is FactValidity.INVALIDATED
            else None
        ),
    )


def _checkpoint(*, complete: bool = True) -> TrajectoryCheckpoint:
    return TrajectoryCheckpoint(
        checkpoint_id="cp2",
        projection_version="deployment-v1",
        state_fingerprint="controller-only-fingerprint",
        environment_facts={"active_color": "blue"},
        requirements={"R-route": False, "R-health": True},
        constraints={"C-availability": True},
        new_evidence=("health probe remained green",),
        actor_action_count=4,
        causal_action_signatures=("set-route:blue",),
        facts=(
            _fact("route-blue", "blue"),
            _fact("route-green-old", "green", validity=FactValidity.INVALIDATED),
        ),
        fact_capture_complete=complete,
    )


def _progress() -> ProgressSnapshot:
    return ProgressSnapshot(
        checkpoint_id="cp2",
        requirements={"R-route": False, "R-health": True},
        constraints={"C-availability": True},
        current_requirement_coverage=0.5,
        best_requirement_coverage=0.5,
        task_progress_delta=0.0,
        gained_requirements=(),
        regressed_requirements=("R-route",),
        new_evidence=("health probe remained green",),
        actor_actions_since_previous=1,
    )


def _frame(
    kind: TrajectoryContextEventKind,
    *,
    turn: int = 2,
    complete: bool = True,
) -> TrajectoryContextFrame:
    event_arguments: dict[str, tuple[str, ...]] = {}
    checkpoint = _checkpoint(complete=complete)
    progress = _progress()
    if kind is TrajectoryContextEventKind.CYCLE_CONFIRMED:
        event_arguments = {
            "causal_actions": ("set-route:green", "set-route:blue"),
            "evidence_refs": ("checkpoint://cp0-cp4",),
        }
    elif kind is TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED:
        event_arguments = {
            "invalidated_fact_ids": ("route-green-old",),
            "evidence_refs": ("deployment://generation/2",),
        }
    elif kind is TrajectoryContextEventKind.CONSTRAINT_VIOLATED:
        event_arguments = {
            "affected_items": ("C-availability",),
            "evidence_refs": ("probe://availability/2",),
        }
        checkpoint = replace(checkpoint, constraints={"C-availability": False})
        progress = replace(progress, constraints={"C-availability": False})
    elif kind is TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED:
        event_arguments = {
            "affected_items": ("R-route",),
            "missing_evidence": ("production route verification",),
        }
    return TrajectoryContextFrame(
        run_id="context-run",
        turn=turn,
        goal="Route production traffic to blue while preserving availability.",
        checkpoint=checkpoint,
        progress=progress,
        event=TrajectoryContextEvent(
            event_id=f"event-{turn}",
            kind=kind,
            **event_arguments,
        ),
        current_plan="Toggle the active route and verify health.",
        refuted_hypotheses=("Changing standby color updates active traffic",),
    )


def _request(turn: int = 2) -> ContextRequest:
    return ContextRequest(
        run_id="context-run",
        source_revision=0,
        turn=turn,
        committed_messages=(SystemMessage("stable"),),
        pending_messages=(UserMessage("deploy"),),
    )


class TrajectoryContextPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def _build(
        self,
        frame: TrajectoryContextFrame | None,
    ) -> ContextView:
        pipeline = TrajectoryContextPipeline(source=lambda request: frame)
        await pipeline.start()
        try:
            return await pipeline.build(
                _request(),
                cancellation=CancellationSource().token,
            )
        finally:
            await pipeline.shutdown()

    async def test_cycle_suspicion_remains_controller_only(self) -> None:
        view = await self._build(_frame(TrajectoryContextEventKind.CYCLE_SUSPECTED))

        self.assertFalse(view.metadata["trajectory_context_visible"])
        self.assertEqual(view.metadata["trajectory_event"], "cycle_suspected")
        self.assertFalse(
            any(isinstance(message, TransientInstruction) for message in view.messages)
        )

    async def test_confirmed_cycle_projects_current_truth_and_provenance(self) -> None:
        view = await self._build(_frame(TrajectoryContextEventKind.CYCLE_CONFIRMED))

        instruction = view.messages[-1]
        self.assertIsInstance(instruction, TransientInstruction)
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(payload["event"], "cycle_confirmed")
        self.assertEqual(
            payload["goal_anchor"],
            _frame(TrajectoryContextEventKind.CYCLE_CONFIRMED).goal,
        )
        self.assertEqual(payload["current_facts"][0]["value"], "blue")
        self.assertEqual(
            payload["current_facts"][0]["evidence_ref"],
            "deployment://evidence/route-blue",
        )
        self.assertEqual(payload["invalidated_facts"], [])
        self.assertNotIn("controller-only-fingerprint", instruction.content)
        self.assertNotIn('"value":"green"', instruction.content)
        self.assertIn("Replan from the Goal", payload["instruction"])

    async def test_external_change_marks_historical_fact_as_invalidated(self) -> None:
        view = await self._build(
            _frame(TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED)
        )

        instruction = view.messages[-1]
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(
            payload["invalidated_facts"],
            [
                {
                    "evidence_ref": "deployment://evidence/route-green-old",
                    "fact_id": "route-green-old",
                    "invalidated_at_checkpoint": "cp2",
                    "predicate": "active_color",
                    "reason": "deployment generation changed",
                    "subject": "deployment/router",
                }
            ],
        )
        self.assertIn("Discard beliefs", payload["instruction"])

    async def test_failed_completion_audit_explicitly_continues_same_run(self) -> None:
        view = await self._build(
            _frame(TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED)
        )

        instruction = view.messages[-1]
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(payload["affected_items"], ["R-route"])
        self.assertEqual(payload["missing_evidence"], ["production route verification"])
        self.assertIn("Continue this Run", payload["instruction"])

    async def test_constraint_violation_projects_a_recovery_boundary(self) -> None:
        view = await self._build(_frame(TrajectoryContextEventKind.CONSTRAINT_VIOLATED))

        instruction = view.messages[-1]
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(payload["constraints"], {"C-availability": False})
        self.assertEqual(payload["affected_items"], ["C-availability"])
        self.assertIn("Recover the violated Constraint", payload["instruction"])

    async def test_incomplete_fact_capture_fails_closed(self) -> None:
        with self.assertRaisesRegex(ContextProtocolError, "complete Fact capture"):
            await self._build(
                _frame(TrajectoryContextEventKind.CYCLE_CONFIRMED, complete=False)
            )


class _TwoTurnModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelResponseCompleted(
                AssistantMessage(tool_calls=(ToolCall("inspect-1", "inspect", {}),))
            )
        else:
            yield ModelResponseCompleted(AssistantMessage(content="replanned"))


class _InspectTool(ToolExecutor):
    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                "inspect",
                "Inspect current deployment State.",
                {"type": "object", "additionalProperties": False},
            ),
        )

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        return ToolExecutionResult({"active_color": "blue"})


class TrajectoryContextRuntimeTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_is_visible_only_at_the_next_model_decision(self) -> None:
        model = _TwoTurnModel()
        frame = _frame(TrajectoryContextEventKind.CYCLE_CONFIRMED)
        pipeline = TrajectoryContextPipeline(
            source=lambda request: frame if request.turn == 2 else None
        )
        kernel = RuntimeKernel(model=model, tools=_InspectTool(), context=pipeline)
        await pipeline.start()
        try:
            outcome = await kernel.run(
                RunSpec(
                    run_id="context-run",
                    base_revision=0,
                    intent=RunIntent.TASK,
                    task="deploy",
                    messages=(SystemMessage("stable"), UserMessage("deploy")),
                    limits=RunLimits(max_turns=2),
                    configuration_revision="trajectory-context-test",
                )
            )
        finally:
            await pipeline.shutdown()

        self.assertTrue(outcome.result.succeeded)
        self.assertEqual(len(model.requests), 2)
        self.assertFalse(
            any(
                isinstance(message, TransientInstruction)
                for message in model.requests[0].messages
            )
        )
        projected = [
            message
            for message in model.requests[1].messages
            if isinstance(message, TransientInstruction)
        ]
        self.assertEqual(len(projected), 1)
        self.assertIn('"event":"cycle_confirmed"', projected[0].content)


if __name__ == "__main__":
    unittest.main()
