#!/usr/bin/env python3
"""PROTOTYPE: replay generated live artifacts through the internal Shadow Analyzer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from ejagent._trajectory import (
    ShadowTrajectoryAnalyzer,
    TrajectoryCheckpoint,
)
from ejagent.contracts import (
    AuditRecord,
    FailureCode,
    RunAudit,
    RunFailure,
    RunPhase,
    RunResult,
    RunStatus,
    RunUsage,
    StopReason,
)


def _verdict(value: object) -> bool | None:
    if value == "pass":
        return True
    if value == "fail":
        return False
    return None


def _run_audit(trial: Mapping[str, object]) -> RunAudit:
    terminal = trial["terminal_result"]
    if not isinstance(terminal, Mapping):
        raise ValueError(f"trial {trial['trial_id']} has no terminal result")
    usage = terminal["usage"]
    if not isinstance(usage, Mapping):
        raise ValueError("terminal usage must be an object")
    result = RunResult(
        run_id=str(terminal["run_id"]),
        status=RunStatus(str(terminal["status"])),
        stop_reason=StopReason(str(terminal["stop_reason"])),
        turns=int(terminal["turns"]),
        output=None if terminal["output"] is None else str(terminal["output"]),
        usage=RunUsage(
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            total_tokens=int(usage["total_tokens"]),
            request_count=int(usage["request_count"]),
            reported_request_count=int(usage["reported_request_count"]),
            cache_read_tokens=usage["cache_read_tokens"],
            cache_write_tokens=usage["cache_write_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
        ),
    )
    records = []
    raw_records = trial["run_audit"]
    if not isinstance(raw_records, list):
        raise ValueError("run_audit must be a list")
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("run_audit entry must be an object")
        payload = raw["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("audit payload must be an object")
        records.append(
            AuditRecord(
                run_id=str(raw["run_id"]),
                sequence=int(raw["sequence"]),
                kind=str(raw["kind"]),
                occurred_at=datetime.fromisoformat(str(raw["occurred_at"])),
                payload=payload,
            )
        )
    failure = None
    raw_failure = trial["failure"]
    if isinstance(raw_failure, Mapping):
        failure = RunFailure(
            phase=RunPhase(str(raw_failure["phase"])),
            code=FailureCode(str(raw_failure["code"])),
            message=str(raw_failure["message"]),
            retryable=bool(raw_failure["retryable"]),
        )
    return RunAudit(
        result=result,
        base_revision=0,
        resulting_revision=0,
        committed=False,
        records=tuple(records),
        failure=failure,
    )


def _checkpoints(trial: Mapping[str, object]) -> tuple[TrajectoryCheckpoint, ...]:
    progress_items = trial["progress"]
    if not isinstance(progress_items, list):
        raise ValueError("progress must be a list")
    progress_by_checkpoint = {
        str(item["checkpoint_id"]): item
        for item in progress_items
        if isinstance(item, Mapping)
    }
    raw_checkpoints = trial["checkpoints"]
    if not isinstance(raw_checkpoints, list):
        raise ValueError("checkpoints must be a list")
    checkpoints: list[TrajectoryCheckpoint] = []
    for raw in raw_checkpoints:
        if not isinstance(raw, Mapping):
            raise ValueError("checkpoint entry must be an object")
        facts = raw["environment_facts"]
        if not isinstance(facts, Mapping):
            raise ValueError("environment_facts must be an object")
        requirements = facts["requirements"]
        constraints = facts["constraints"]
        if not isinstance(requirements, Mapping) or not isinstance(
            constraints, Mapping
        ):
            raise ValueError("requirement and constraint Facts must be objects")
        checkpoint_id = str(raw["checkpoint_id"])
        progress = progress_by_checkpoint[checkpoint_id]
        new_evidence = progress["new_evidence"]
        if not isinstance(new_evidence, list):
            raise ValueError("new_evidence must be a list")
        causal_signatures = raw.get("causal_action_signatures", ())
        if not isinstance(causal_signatures, (list, tuple)):
            raise ValueError("causal_action_signatures must be a list")
        checkpoints.append(
            TrajectoryCheckpoint(
                checkpoint_id=checkpoint_id,
                projection_version=str(facts["fixture_version"]),
                state_fingerprint=str(raw["state_fingerprint"]),
                environment_facts=facts,
                requirements={
                    str(name): _verdict(value) for name, value in requirements.items()
                },
                constraints={
                    str(name): _verdict(value) for name, value in constraints.items()
                },
                new_evidence=tuple(str(item) for item in new_evidence),
                actor_action_count=int(raw["actor_action_count"]),
                causal_action_signatures=tuple(str(item) for item in causal_signatures),
                causally_complete=bool(raw["causally_complete"]),
            )
        )
    return tuple(checkpoints)


def replay(path: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    trials = artifact["trials"]
    if not isinstance(trials, list):
        raise ValueError("artifact trials must be a list")
    analyzer = ShadowTrajectoryAnalyzer(max_period=3)
    results = []
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise ValueError("trial must be an object")
        report = analyzer.analyze(_run_audit(trial), _checkpoints(trial))
        old_recurrence = trial["recurrence"]
        if not isinstance(old_recurrence, Mapping):
            raise ValueError("trial recurrence must be an object")
        expected = str(old_recurrence["verdict"])
        matched = report.verdict.value == expected
        results.append(
            {
                "trial_id": trial["trial_id"],
                "expected_recurrence": expected,
                "shadow_verdict": report.verdict.value,
                "matched": matched,
                "normalized_action_count": len(report.actions),
                "normalized_observation_count": len(report.observations),
                "diagnostics": list(report.diagnostics),
            }
        )
    return {
        "artifact": str(path),
        "trial_count": len(results),
        "all_recurrence_verdicts_matched": all(item["matched"] for item in results),
        "all_audits_normalized_without_diagnostics": all(
            not item["diagnostics"] for item in results
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = replay(args.artifact)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return (
        0
        if (
            report["all_recurrence_verdicts_matched"]
            and report["all_audits_normalized_without_diagnostics"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
