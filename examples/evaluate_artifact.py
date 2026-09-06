"""Evaluate a UTF-8 JSON artifact locally, without any model or credentials."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ejagent.contracts import CancellationSource, CompletionCandidate
from ejagent.evaluation import (
    EvaluationCriterion,
    EvaluationMonitor,
    EvaluationPlan,
    FileEvidenceSource,
    GoalEvaluator,
    JsonlEvaluationJournal,
    file_exists,
    json_fields,
)
from ejagent.kernel.trajectory import (
    CheckpointSignal,
    CheckpointTrigger,
    TrajectoryCost,
)


async def evaluate(path: Path, journal: Path | None) -> None:
    plan = EvaluationPlan(
        goal="Produce a JSON artifact containing an answer field",
        version="artifact.v1",
        requirements=(
            EvaluationCriterion("exists", "Artifact exists", "exists", ("artifact",)),
            EvaluationCriterion(
                "shape", "JSON object contains answer", "shape", ("artifact",)
            ),
        ),
        artifact_refs=(str(path),),
    )
    evaluator = GoalEvaluator(
        sources={"artifact": FileEvidenceSource(path)},
        verifiers={"exists": file_exists, "shape": json_fields("answer")},
        report_sink=JsonlEvaluationJournal(journal) if journal else None,
    )
    monitor = EvaluationMonitor(evaluator)
    token = CancellationSource().token
    try:
        await monitor.capture(
            CheckpointSignal(
                "artifact-demo",
                CheckpointTrigger.BASELINE,
                0,
                TrajectoryCost(),
                evaluation_plan=plan,
            ),
            cancellation=token,
        )
        receipt = await monitor.capture(
            CheckpointSignal(
                "artifact-demo",
                CheckpointTrigger.COMPLETION_PROPOSED,
                1,
                TrajectoryCost(),
                evaluation_plan=plan,
                completion_candidate=CompletionCandidate("Artifact is ready."),
            ),
            cancellation=token,
        )
        report = evaluator.latest_report("artifact-demo")
        assert report is not None
        for item in report.requirements:
            print(f"{item.criterion_id}: {item.status.value} — {item.rationale}")
        print(f"Completion advice: {receipt.completion_allowed}")
        print(f"Evaluation cost: {report.cost}")
    finally:
        monitor.close_run("artifact-demo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--journal", type=Path)
    arguments = parser.parse_args()
    asyncio.run(evaluate(arguments.path, arguments.journal))
