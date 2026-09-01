#!/usr/bin/env python3
"""PROTOTYPE trajectory experiments; not production Runtime code."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ejagent._trajectory import (  # noqa: E402
    ShadowTrajectoryAnalyzer,
    TrajectoryCheckpoint,
)
from ejagent.contracts import (  # noqa: E402
    AssistantMessage,
    CancellationToken,
    ContextRequest,
    ContextView,
    ModelRequest,
    ModelResponseCompleted,
    ModelUsage,
    RunAudit,
    RunIntent,
    RunLimits,
    RunOutcome,
    RunSpec,
    RunStatus,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    TransientInstruction,
)
from ejagent.kernel import RuntimeKernel  # noqa: E402
from ejagent.providers import ModelConfig, OpenAIModelPort  # noqa: E402
from ejagent.tools import FunctionTool, FunctionToolExecutor  # noqa: E402

EXPERIMENT_ROOT = Path(__file__).resolve().parent
FS001_ROOT = EXPERIMENT_ROOT / "fs001"
BASELINE_ROOT = FS001_ROOT / "baseline"
GOLD_ROOT = FS001_ROOT / "gold" / "auth_fixture"
FIXTURE_MANIFEST_PATH = BASELINE_ROOT / "fixture-manifest.json"
LIVE_PREREGISTRATION_PATH = FS001_ROOT / "live-preregistration.json"

VERIFIERS = {
    "R1": "tests.test_r1_expired_access",
    "R2": "tests.test_r2_refresh_flow",
    "C1": "tests.test_c1_public_compatibility",
}


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _file_digest(path)
        for path in sorted((root / "auth_fixture").glob("*.py"))
    }


def _validate_fixture_manifest() -> dict[str, object]:
    manifest = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_files = manifest["files"]
    expected_gold = manifest["gold_files"]
    assert isinstance(expected_files, dict)
    assert isinstance(expected_gold, dict)

    discovered_files = {
        str(path.relative_to(BASELINE_ROOT))
        for path in BASELINE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    actual_files = {
        relative_path: _file_digest(BASELINE_ROOT / relative_path)
        for relative_path in expected_files
    }
    actual_gold = {
        relative_path: _file_digest(FS001_ROOT / "gold" / relative_path)
        for relative_path in expected_gold
    }
    file_set_matches = discovered_files == set(expected_files)
    hashes_match = actual_files == expected_files
    gold_hashes_match = actual_gold == expected_gold
    return {
        "fixture_version": manifest["fixture_version"],
        "file_set_matches": file_set_matches,
        "hashes_match": hashes_match,
        "gold_hashes_match": gold_hashes_match,
        "valid": file_set_matches and hashes_match and gold_hashes_match,
    }


def _sanitized_output(output: str, root: Path) -> str:
    value = output.replace(str(root), "<fixture>")
    value = re.sub(r"Ran (\d+) tests? in [0-9.]+s", r"Ran \1 test(s)", value)
    return "\n".join(line.rstrip() for line in value.splitlines() if line.strip())


def _run_command(root: Path, command: list[str]) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    output = _sanitized_output(completed.stdout + completed.stderr, root)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "verdict": "pass" if completed.returncode == 0 else "fail",
        "output": output,
        "failure_signature": None if completed.returncode == 0 else _digest(output),
    }


def _run_verifier(root: Path, name: str) -> dict[str, object]:
    if name == "ALL":
        return {key: _run_verifier(root, key) for key in VERIFIERS}
    try:
        module = VERIFIERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown verifier {name!r}") from exc
    return _run_command(root, [sys.executable, "-m", "unittest", module, "-v"])


def _probe_contract(root: Path) -> dict[str, object]:
    result = _run_command(root, [sys.executable, "probe.py"])
    if result["exit_code"] != 0:
        return {
            "verdict": "fail",
            "data": None,
            "output": result["output"],
            "failure_signature": result["failure_signature"],
        }
    output = str(result["output"]).splitlines()[-1]
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return {
            "verdict": "fail",
            "data": None,
            "output": result["output"],
            "failure_signature": _digest(result["output"]),
        }
    return {
        "verdict": "pass",
        "data": data,
        "output": result["output"],
        "failure_signature": None,
    }


def _copy_baseline(destination: Path) -> Path:
    root = destination / "fixture"
    shutil.copytree(
        BASELINE_ROOT,
        root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return root


def _write_validation_policy(root: Path, allow_expired: bool) -> None:
    path = root / "auth_fixture" / "validation.py"
    content = path.read_text(encoding="utf-8")
    content, replacements = re.subn(
        r"ALLOW_EXPIRED_ACCESS: Final\[bool\] = (True|False)",
        f"ALLOW_EXPIRED_ACCESS: Final[bool] = {allow_expired}",
        content,
    )
    if replacements != 1:
        raise RuntimeError("fixture validation policy marker was not unique")
    path.write_text(content, encoding="utf-8")


def _apply_gold(root: Path) -> None:
    for name in ("api.py", "validation.py"):
        shutil.copyfile(GOLD_ROOT / name, root / "auth_fixture" / name)


def _verdicts(verifiers: Mapping[str, object]) -> dict[str, str]:
    return {
        name: str(item["verdict"])
        for name, item in verifiers.items()
        if isinstance(item, Mapping)
    }


def _checkpoint(
    root: Path,
    *,
    trial_id: str,
    sequence: int,
    causal_actions: list[str],
    actor_action_count: int,
    causal_action_signatures: list[str] | None = None,
) -> dict[str, object]:
    manifest = _source_manifest(root)
    verifiers = _run_verifier(root, "ALL")
    assert isinstance(verifiers, dict)
    probe_evidence = _probe_contract(root)
    probe = probe_evidence["data"]
    if not isinstance(probe, Mapping):
        probe = {}
    verdicts = _verdicts(verifiers)
    facts = {
        "fixture_version": "fs001-v1",
        "source_manifest_hash": _digest(manifest),
        "validation_file_hash": manifest["auth_fixture/validation.py"],
        "public_signature_hash": _digest(
            {
                "access": probe.get("access_signature"),
                "refresh": probe.get("refresh_signature"),
            }
        ),
        "token_schema_hash": _digest(probe.get("token_payload_keys")),
        "requirements": {"R1": verdicts["R1"], "R2": verdicts["R2"]},
        "constraints": {"C1": verdicts["C1"]},
        "failure_signatures": {
            name: item["failure_signature"]
            for name, item in verifiers.items()
            if isinstance(item, Mapping)
        },
        "external_side_effect_count": 0,
    }
    fingerprint = _digest(facts)
    return {
        "checkpoint_id": f"{trial_id}:cp{sequence}",
        "trial_id": trial_id,
        "sequence": sequence,
        "causal_action_ids": list(causal_actions),
        "causal_action_signatures": list(causal_action_signatures or ()),
        "actor_action_count": actor_action_count,
        "source_manifest": manifest,
        "environment_facts": facts,
        "state_fingerprint": fingerprint,
        "causally_complete": True,
        "verifier_evidence": verifiers,
        "contract_probe": probe_evidence,
    }


def _coverage(checkpoint: Mapping[str, object]) -> float:
    facts = checkpoint["environment_facts"]
    assert isinstance(facts, Mapping)
    requirements = facts["requirements"]
    assert isinstance(requirements, Mapping)
    return sum(value == "pass" for value in requirements.values()) / len(requirements)


def _progress_records(checkpoints: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_states: set[str] = set()
    best = 0.0
    previous_coverage: float | None = None
    previous_requirements: Mapping[str, object] | None = None
    previous_action_count = 0
    for checkpoint in checkpoints:
        facts = checkpoint["environment_facts"]
        assert isinstance(facts, Mapping)
        requirements = facts["requirements"]
        constraints = facts["constraints"]
        assert isinstance(requirements, Mapping)
        assert isinstance(constraints, Mapping)
        coverage = _coverage(checkpoint)
        gained: list[str] = []
        regressed: list[str] = []
        if previous_requirements is not None:
            gained = [
                key
                for key, value in requirements.items()
                if value == "pass" and previous_requirements.get(key) != "pass"
            ]
            regressed = [
                key
                for key, value in requirements.items()
                if value != "pass" and previous_requirements.get(key) == "pass"
            ]
        fingerprint = str(checkpoint["state_fingerprint"])
        new_evidence = []
        if records and fingerprint not in seen_states:
            new_evidence.append(f"previously unseen verified state {fingerprint[:12]}")
        action_count = int(checkpoint["actor_action_count"])
        best = max(best, coverage)
        records.append(
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "requirements": dict(requirements),
                "constraints": dict(constraints),
                "current_requirement_coverage": coverage,
                "best_requirement_coverage": best,
                "task_progress_delta": (
                    0.0 if previous_coverage is None else coverage - previous_coverage
                ),
                "gained_requirements": gained,
                "regressed_requirements": regressed,
                "new_evidence": new_evidence,
                "blockers": [],
                "cost_since_previous_checkpoint": {
                    "actor_actions": action_count - previous_action_count
                },
                "assessment_source": "fs001 deterministic verifier",
            }
        )
        seen_states.add(fingerprint)
        previous_coverage = coverage
        previous_requirements = requirements
        previous_action_count = action_count
    return records


def _facts_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return left["environment_facts"] == right["environment_facts"]


def _detect_periodic_non_progress(
    checkpoints: list[Mapping[str, object]],
    progress: list[Mapping[str, object]],
    *,
    max_period: int = 3,
) -> dict[str, object]:
    fingerprints = [str(item["state_fingerprint"]) for item in checkpoints]
    action_signatures = [
        tuple(str(value) for value in item.get("causal_action_signatures", ()))
        for item in checkpoints[1:]
    ]

    # Confirmation requires two complete transition cycles. For period two,
    # S0/A/S1/B/S0/A/S1 is only a candidate; the final B/S0 closes the second
    # action-and-state path.
    for period in range(1, max_period + 1):
        window_size = period * 2 + 1
        if len(fingerprints) < window_size:
            continue
        state_window = fingerprints[-window_size:]
        if state_window[: period + 1] != state_window[period:]:
            continue
        action_window = action_signatures[-period * 2 :]
        if (
            len(action_window) != period * 2
            or not all(action_window)
            or action_window[:period] != action_window[period:]
        ):
            continue
        repeated_progress = progress[-period:]
        task_progress = sum(
            float(item.get("task_progress_delta", 0.0)) for item in repeated_progress
        )
        no_task_progress = task_progress <= 0
        no_new_evidence = all(not item["new_evidence"] for item in repeated_progress)
        costs_increased = any(
            int(item["cost_since_previous_checkpoint"]["actor_actions"]) > 0
            for item in repeated_progress
        )
        candidate_checkpoints = checkpoints[-window_size:]
        facts_equivalent = all(
            _facts_equal(
                candidate_checkpoints[offset],
                candidate_checkpoints[period + offset],
            )
            for offset in range(period + 1)
        )
        causally_complete = all(
            bool(item.get("causally_complete", False)) for item in candidate_checkpoints
        )
        if (
            facts_equivalent
            and causally_complete
            and no_task_progress
            and no_new_evidence
            and costs_increased
        ):
            return {
                "verdict": "non_progress_cycle",
                "period": period,
                "candidate_checkpoint_ids": [
                    item["checkpoint_id"] for item in candidate_checkpoints
                ],
                "fingerprints_equal": True,
                "underlying_facts_equivalent": True,
                "repeated_action_path": True,
                "new_evidence_after_first_cycle": False,
                "task_progress_over_repeated_window": task_progress,
                "cost_increased": True,
            }

    for period in range(1, max_period + 1):
        window_size = period * 2
        if len(fingerprints) < window_size:
            continue
        candidate_checkpoints = checkpoints[-window_size:]
        if fingerprints[-window_size:-period] != fingerprints[-period:]:
            continue
        repeated_progress = progress[-period:]
        task_progress = sum(
            float(item.get("task_progress_delta", 0.0)) for item in repeated_progress
        )
        facts_equivalent = all(
            _facts_equal(
                candidate_checkpoints[offset],
                candidate_checkpoints[period + offset],
            )
            for offset in range(period)
        )
        if (
            facts_equivalent
            and all(
                bool(item.get("causally_complete", False))
                for item in candidate_checkpoints
            )
            and task_progress <= 0
            and all(not item["new_evidence"] for item in repeated_progress)
            and any(
                int(item["cost_since_previous_checkpoint"]["actor_actions"]) > 0
                for item in repeated_progress
            )
        ):
            return {
                "verdict": "cycle_suspected",
                "period": period,
                "candidate_checkpoint_ids": [
                    item["checkpoint_id"] for item in candidate_checkpoints
                ],
                "fingerprints_equal": True,
                "underlying_facts_equivalent": True,
                "repeated_action_path": False,
                "new_evidence_after_first_cycle": False,
                "task_progress_over_repeated_window": task_progress,
                "cost_increased": True,
            }
    return {"verdict": "no_cycle", "period": None}


def _expected_state(
    checkpoint: Mapping[str, object], expected: tuple[str, str, str]
) -> bool:
    facts = checkpoint["environment_facts"]
    assert isinstance(facts, Mapping)
    requirements = facts["requirements"]
    constraints = facts["constraints"]
    assert isinstance(requirements, Mapping)
    assert isinstance(constraints, Mapping)
    return (
        requirements["R1"],
        requirements["R2"],
        constraints["C1"],
    ) == expected


class _ReplayModel:
    def __init__(self, strict_content: str, legacy_content: str) -> None:
        self._responses = (
            (
                "write_file",
                {"path": "auth_fixture/validation.py", "content": strict_content},
            ),
            ("run_verifier", {"name": "ALL"}),
            (
                "write_file",
                {"path": "auth_fixture/validation.py", "content": legacy_content},
            ),
            ("run_verifier", {"name": "ALL"}),
            (
                "write_file",
                {"path": "auth_fixture/validation.py", "content": strict_content},
            ),
            ("run_verifier", {"name": "ALL"}),
            (
                "write_file",
                {"path": "auth_fixture/validation.py", "content": legacy_content},
            ),
            ("run_verifier", {"name": "ALL"}),
        )
        self._index = 0
        self.request_trace: list[dict[str, object]] = []

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelResponseCompleted]:
        cancellation.raise_if_cancelled()
        transient_messages = [
            message
            for message in request.messages
            if isinstance(message, TransientInstruction)
        ]
        self.request_trace.append(
            {
                "turn": self._index + 1,
                "transient_sources": [message.source for message in transient_messages],
                "state_projection_visible": any(
                    message.source == "fs001_experiment_state_projection"
                    for message in transient_messages
                ),
                "cycle_confirmed_visible": any(
                    '"event":"CycleConfirmed"' in message.content
                    for message in transient_messages
                ),
            }
        )
        tool_name, arguments = self._responses[self._index % len(self._responses)]
        self._index += 1
        yield ModelResponseCompleted(
            AssistantMessage(
                tool_calls=(
                    ToolCall(
                        id=f"replay-call-{self._index}",
                        name=tool_name,
                        arguments=arguments,
                    ),
                )
            ),
            ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )


@dataclass
class _FixtureToolState:
    root: Path
    trial_id: str
    checkpoints: list[dict[str, object]]
    actor_action_count: int = 0
    actions: list[dict[str, object]] = field(default_factory=list)
    observations: list[dict[str, object]] = field(default_factory=list)

    def capture_after_mutation(self, call: ToolCall) -> None:
        self.checkpoints.append(
            _checkpoint(
                self.root,
                trial_id=self.trial_id,
                sequence=len(self.checkpoints),
                causal_actions=[call.id],
                actor_action_count=self.actor_action_count,
                causal_action_signatures=[
                    _digest(
                        {
                            "tool_name": call.name,
                            "arguments": _action_arguments(call),
                        }
                    )
                ],
            )
        )


def _fixture_tools(state: _FixtureToolState) -> FunctionToolExecutor:
    async def write_file(
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        state.actor_action_count += 1
        path = call.arguments.get("path")
        content = call.arguments.get("content")
        if path != "auth_fixture/validation.py" or not isinstance(content, str):
            return ToolExecutionResult(
                {"status": "denied", "reason": "write outside bounded fixture"},
                error="write outside bounded fixture",
            )
        target = state.root / path
        target.write_text(content, encoding="utf-8")
        state.capture_after_mutation(call)
        return ToolExecutionResult({"status": "written", "path": path})

    async def run_verifier(
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        state.actor_action_count += 1
        name = call.arguments.get("name")
        if name not in (*VERIFIERS, "ALL"):
            return ToolExecutionResult(
                {"status": "error", "reason": "unknown verifier"},
                error="unknown verifier",
            )
        return ToolExecutionResult(_run_verifier(state.root, str(name)))

    return FunctionToolExecutor(
        (
            FunctionTool(
                ToolDefinition(
                    "write_file",
                    "Replace one bounded fixture file with complete text.",
                    {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                ),
                write_file,
            ),
            FunctionTool(
                ToolDefinition(
                    "run_verifier",
                    "Run a focused or complete frozen verifier.",
                    {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": [*VERIFIERS, "ALL"],
                            }
                        },
                        "required": ["name"],
                    },
                ),
                run_verifier,
            ),
        )
    )


READABLE_FIXTURE_PATHS = (
    "auth_fixture/__init__.py",
    "auth_fixture/api.py",
    "auth_fixture/tokens.py",
    "auth_fixture/validation.py",
    "tests/test_r1_expired_access.py",
    "tests/test_r2_refresh_flow.py",
    "tests/test_c1_public_compatibility.py",
    "probe.py",
)
WRITABLE_FIXTURE_PATHS = (
    "auth_fixture/__init__.py",
    "auth_fixture/api.py",
    "auth_fixture/tokens.py",
    "auth_fixture/validation.py",
)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _action_arguments(call: ToolCall) -> dict[str, object]:
    arguments = dict(call.arguments)
    content = arguments.pop("content", None)
    if isinstance(content, str):
        arguments["content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        arguments["content_bytes"] = len(content.encode())
    return arguments


def _begin_action(
    state: _FixtureToolState,
    call: ToolCall,
    *,
    side_effect_scope: str,
) -> dict[str, object]:
    state.actor_action_count += 1
    normalized_arguments = _action_arguments(call)
    action = {
        "action_id": f"{state.trial_id}:action:{len(state.actions) + 1}",
        "actor_or_evaluator": "actor",
        "model_turn": None,
        "tool_call_id": call.id,
        "tool_name": call.name,
        "normalized_arguments": normalized_arguments,
        "action_signature": _digest(
            {"tool_name": call.name, "arguments": normalized_arguments}
        ),
        "proposed_at": None,
        "started_at": _utc_timestamp(),
        "completed_at": None,
        "batch_id": None,
        "side_effect_scope": side_effect_scope,
        "result_reference": None,
        "result_signature": None,
        "is_error": False,
    }
    state.actions.append(action)
    return action


def _finish_action(
    state: _FixtureToolState,
    action: dict[str, object],
    result: object,
    *,
    is_error: bool = False,
) -> None:
    action["completed_at"] = _utc_timestamp()
    action["result_signature"] = _digest(result)
    action["is_error"] = is_error
    observation_id = f"{state.trial_id}:observation:{len(state.observations) + 1}"
    action["result_reference"] = observation_id
    action_id = str(action["action_id"])
    state.observations.append(
        {
            "observation_id": observation_id,
            "producer": "tool",
            "action_id": action_id,
            "raw_artifact_reference": f"inline:{action_id}",
            "normalized_summary": result,
            "captured_at": action["completed_at"],
            "source_checkpoint": (
                state.checkpoints[-1]["checkpoint_id"] if state.checkpoints else None
            ),
            "sensitivity_classification": "fixture_only",
        }
    )


def _bounded_diff(root: Path) -> str:
    lines: list[str] = []
    for relative_path in WRITABLE_FIXTURE_PATHS:
        baseline = (BASELINE_ROOT / relative_path).read_text(encoding="utf-8")
        current = (root / relative_path).read_text(encoding="utf-8")
        lines.extend(
            difflib.unified_diff(
                baseline.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"baseline/{relative_path}",
                tofile=f"current/{relative_path}",
            )
        )
    value = "".join(lines)
    if len(value) > 20_000:
        return value[:20_000] + "\n[diff truncated at 20,000 characters]\n"
    return value or "No source changes."


def _live_fixture_tools(state: _FixtureToolState) -> FunctionToolExecutor:
    async def read_file(
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        action = _begin_action(state, call, side_effect_scope="none")
        path = call.arguments.get("path")
        if path not in READABLE_FIXTURE_PATHS:
            result = {"status": "denied", "reason": "path is outside readable fixture"}
            _finish_action(state, action, result, is_error=True)
            return ToolExecutionResult(result, error=str(result["reason"]))
        content = (state.root / str(path)).read_text(encoding="utf-8")
        result = {
            "status": "read",
            "path": path,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "content": content,
        }
        _finish_action(state, action, result)
        return ToolExecutionResult(result)

    async def write_file(
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        action = _begin_action(state, call, side_effect_scope="fixture_source")
        path = call.arguments.get("path")
        content = call.arguments.get("content")
        if path not in WRITABLE_FIXTURE_PATHS or not isinstance(content, str):
            result = {"status": "denied", "reason": "invalid bounded source write"}
            _finish_action(state, action, result, is_error=True)
            return ToolExecutionResult(result, error=str(result["reason"]))
        target = state.root / str(path)
        target.write_text(content, encoding="utf-8")
        state.capture_after_mutation(call)
        result = {
            "status": "written",
            "path": path,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "evaluator_checkpoint": "captured_out_of_band",
        }
        _finish_action(state, action, result)
        return ToolExecutionResult(result)

    async def run_verifier(
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        action = _begin_action(state, call, side_effect_scope="none")
        name = call.arguments.get("name")
        if name not in (*VERIFIERS, "ALL"):
            result = {"status": "error", "reason": "unknown verifier"}
            _finish_action(state, action, result, is_error=True)
            return ToolExecutionResult(result, error=str(result["reason"]))
        result = _run_verifier(state.root, str(name))
        _finish_action(state, action, result)
        return ToolExecutionResult(result)

    async def inspect_diff(
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        action = _begin_action(state, call, side_effect_scope="none")
        result = {"status": "diff", "content": _bounded_diff(state.root)}
        _finish_action(state, action, result)
        return ToolExecutionResult(result)

    return FunctionToolExecutor(
        (
            FunctionTool(
                ToolDefinition(
                    "read_file",
                    "Read one allowed fixture source, test, or probe file.",
                    {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "enum": list(READABLE_FIXTURE_PATHS),
                            }
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
                read_file,
            ),
            FunctionTool(
                ToolDefinition(
                    "write_file",
                    "Replace the complete contents of one allowed fixture source file.",
                    {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "enum": list(WRITABLE_FIXTURE_PATHS),
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                ),
                write_file,
            ),
            FunctionTool(
                ToolDefinition(
                    "run_verifier",
                    "Run one focused verifier or the complete frozen verifier set.",
                    {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": [*VERIFIERS, "ALL"],
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                ),
                run_verifier,
            ),
            FunctionTool(
                ToolDefinition(
                    "inspect_diff",
                    "Inspect the bounded source diff from the frozen fixture baseline.",
                    {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
                inspect_diff,
            ),
        )
    )


class _ExperimentContextPipeline:
    def __init__(self, state: _FixtureToolState, condition: str) -> None:
        self._state = state
        self._condition = condition

    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        cancellation.raise_if_cancelled()
        checkpoint = self._state.checkpoints[-1]
        progress = _progress_records(self._state.checkpoints)[-1]
        facts = checkpoint["environment_facts"]
        assert isinstance(facts, Mapping)
        projection: dict[str, object] = {
            "projection": "experiment_owned_current_state",
            "checkpoint": checkpoint["checkpoint_id"],
            "provenance": "FS-001 frozen deterministic V-ALL evaluator",
            "requirements": facts["requirements"],
            "constraints": facts["constraints"],
            "current_requirement_coverage": progress["current_requirement_coverage"],
            "best_requirement_coverage": progress["best_requirement_coverage"],
            "task_progress_delta": progress["task_progress_delta"],
            "regressed_requirements": progress["regressed_requirements"],
            "new_evidence": progress["new_evidence"],
            "note": "Historical tool results may be stale after a source mutation.",
        }
        recurrence = _detect_periodic_non_progress(
            self._state.checkpoints,
            _progress_records(self._state.checkpoints),
        )
        if (
            self._condition == "confirmed_cycle_intervention"
            and recurrence["verdict"] == "non_progress_cycle"
        ):
            projection["intervention"] = {
                "event": "CycleConfirmed",
                "checkpoint_ids": recurrence["candidate_checkpoint_ids"],
                "assessment": (
                    "Equivalent states and the same no-progress path have repeated "
                    "without new evidence while action cost increased. Replan from the "
                    "original goal and current facts; do not repeat the exhausted path."
                ),
            }
        message = TransientInstruction(
            _stable_json(projection),
            "fs001_experiment_state_projection",
        )
        return ContextView(
            run_id=request.run_id,
            source_revision=request.source_revision,
            turn=request.turn,
            messages=(*request.messages, message),
            metadata={
                **request.metadata,
                "projection": self._condition,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "cycle_intervention": "intervention" in projection,
            },
        )


async def _deterministic_replay(root: Path) -> dict[str, object]:
    trial_id = "fs001-deterministic-replay"
    baseline_validation = (root / "auth_fixture" / "validation.py").read_text(
        encoding="utf-8"
    )
    strict_validation = baseline_validation.replace(
        "ALLOW_EXPIRED_ACCESS: Final[bool] = True",
        "ALLOW_EXPIRED_ACCESS: Final[bool] = False",
    )
    state = _FixtureToolState(root=root, trial_id=trial_id, checkpoints=[])
    state.checkpoints.append(
        _checkpoint(
            root,
            trial_id=trial_id,
            sequence=0,
            causal_actions=[],
            actor_action_count=0,
        )
    )
    model = _ReplayModel(strict_validation, baseline_validation)
    kernel = RuntimeKernel(model=model, tools=_fixture_tools(state))
    outcome = await kernel.run(
        RunSpec(
            run_id=trial_id,
            base_revision=0,
            intent=RunIntent.TASK,
            task="Fix expired access rejection without breaking refresh compatibility.",
            messages=(SystemMessage("FS-001 deterministic replay."),),
            limits=RunLimits(max_turns=8, max_repeated_tool_calls=3),
            configuration_revision="fs001-v1",
        )
    )
    progress = _progress_records(state.checkpoints)
    recurrence = _detect_periodic_non_progress(state.checkpoints, progress)
    shadow_report = _shadow_analyze(outcome, state.checkpoints, progress)
    return {
        "trial_id": trial_id,
        "outcome": {
            "status": outcome.result.status.value,
            "stop_reason": outcome.result.stop_reason.value,
            "turns": outcome.result.turns,
            "audit_kinds": [record.kind for record in outcome.audit_records],
            "tool_started_count": sum(
                record.kind == "tool_started" for record in outcome.audit_records
            ),
        },
        "checkpoints": state.checkpoints,
        "progress": progress,
        "recurrence": recurrence,
        "shadow_report": shadow_report,
        "request_trace": model.request_trace,
        "expected": {
            "runtime_status": RunStatus.FAILED.value,
            "runtime_stop_reason": StopReason.MAX_STEPS.value,
            "offline_classification": "non_progress_cycle",
        },
    }


def _checkpoint_verdict(value: object) -> bool | None:
    if value == "pass":
        return True
    if value == "fail":
        return False
    return None


def _shadow_analyze(
    outcome: RunOutcome,
    checkpoints: list[dict[str, object]],
    progress: list[dict[str, object]],
) -> dict[str, object]:
    typed_checkpoints: list[TrajectoryCheckpoint] = []
    for checkpoint, snapshot in zip(checkpoints, progress, strict=True):
        facts = checkpoint["environment_facts"]
        assert isinstance(facts, Mapping)
        requirements = facts["requirements"]
        constraints = facts["constraints"]
        assert isinstance(requirements, Mapping)
        assert isinstance(constraints, Mapping)
        typed_checkpoints.append(
            TrajectoryCheckpoint(
                checkpoint_id=str(checkpoint["checkpoint_id"]),
                projection_version=str(facts["fixture_version"]),
                state_fingerprint=str(checkpoint["state_fingerprint"]),
                environment_facts=facts,
                requirements={
                    str(name): _checkpoint_verdict(value)
                    for name, value in requirements.items()
                },
                constraints={
                    str(name): _checkpoint_verdict(value)
                    for name, value in constraints.items()
                },
                new_evidence=tuple(str(item) for item in snapshot["new_evidence"]),
                actor_action_count=int(checkpoint["actor_action_count"]),
                causal_action_signatures=tuple(
                    str(item) for item in checkpoint.get("causal_action_signatures", ())
                ),
                causally_complete=bool(checkpoint["causally_complete"]),
            )
        )
    audit = RunAudit(
        result=outcome.result,
        base_revision=0,
        resulting_revision=0,
        committed=False,
        records=outcome.audit_records,
        failure=outcome.failure,
    )
    report = ShadowTrajectoryAnalyzer(max_period=3).analyze(audit, typed_checkpoints)
    return {
        "run_id": report.run_id,
        "verdict": report.verdict.value,
        "period": report.period,
        "candidate_checkpoint_ids": list(report.candidate_checkpoint_ids),
        "repeated_action_path": report.repeated_action_path,
        "task_progress_over_repeated_window": (
            report.task_progress_over_repeated_window
        ),
        "normalized_action_count": len(report.actions),
        "normalized_observation_count": len(report.observations),
        "diagnostics": list(report.diagnostics),
    }


async def _context_timing_controls() -> dict[str, object]:
    results: dict[str, object] = {}
    for condition in (
        "baseline_identity",
        "state_progress_projection",
        "confirmed_cycle_intervention",
    ):
        with tempfile.TemporaryDirectory(prefix=f"ejagent-context-{condition}-") as tmp:
            root = _copy_baseline(Path(tmp))
            trial_id = f"fs001-context-timing:{condition}"
            baseline_validation = (root / "auth_fixture" / "validation.py").read_text(
                encoding="utf-8"
            )
            strict_validation = baseline_validation.replace(
                "ALLOW_EXPIRED_ACCESS: Final[bool] = True",
                "ALLOW_EXPIRED_ACCESS: Final[bool] = False",
            )
            state = _FixtureToolState(root=root, trial_id=trial_id, checkpoints=[])
            state.checkpoints.append(
                _checkpoint(
                    root,
                    trial_id=trial_id,
                    sequence=0,
                    causal_actions=[],
                    actor_action_count=0,
                )
            )
            model = _ReplayModel(strict_validation, baseline_validation)
            context = (
                None
                if condition == "baseline_identity"
                else _ExperimentContextPipeline(state, condition)
            )
            kernel = RuntimeKernel(
                model=model,
                tools=_fixture_tools(state),
                context=context,
            )
            await kernel.run(
                RunSpec(
                    run_id=trial_id,
                    base_revision=0,
                    intent=RunIntent.TASK,
                    task="FS-001 context timing control.",
                    messages=(SystemMessage("FS-001 context timing control."),),
                    limits=RunLimits(max_turns=8, max_repeated_tool_calls=3),
                    configuration_revision="fs001-context-v1",
                )
            )
            projection_turns = [
                item["turn"]
                for item in model.request_trace
                if item["state_projection_visible"]
            ]
            cycle_turns = [
                item["turn"]
                for item in model.request_trace
                if item["cycle_confirmed_visible"]
            ]
            results[condition] = {
                "request_trace": model.request_trace,
                "projection_turns": projection_turns,
                "cycle_intervention_turns": cycle_turns,
            }
    baseline_ok = not results["baseline_identity"]["projection_turns"]
    projection_ok = (
        results["state_progress_projection"]["projection_turns"] == list(range(1, 9))
        and not results["state_progress_projection"]["cycle_intervention_turns"]
    )
    cycle_turns = results["confirmed_cycle_intervention"]["cycle_intervention_turns"]
    cycle_timing_ok = cycle_turns == [8]
    return {
        "conditions": results,
        "expected": {
            "baseline_has_no_projection": baseline_ok,
            "state_projection_on_every_decision": projection_ok,
            "first_confirmed_cycle_intervention_turn": 8,
            "cycle_intervention_timing_matches": cycle_timing_ok,
        },
        "passed": baseline_ok and projection_ok and cycle_timing_ok,
    }


def _gold_controls() -> dict[str, object]:
    results: dict[str, object] = {}
    for source_state in ("S0", "S1"):
        with tempfile.TemporaryDirectory(
            prefix=f"ejagent-fs001-gold-{source_state}-"
        ) as tmp:
            root = _copy_baseline(Path(tmp))
            if source_state == "S1":
                _write_validation_policy(root, allow_expired=False)
            before = _checkpoint(
                root,
                trial_id=f"gold-from-{source_state}",
                sequence=0,
                causal_actions=[],
                actor_action_count=0,
            )
            _apply_gold(root)
            after = _checkpoint(
                root,
                trial_id=f"gold-from-{source_state}",
                sequence=1,
                causal_actions=["gold-action-c"],
                actor_action_count=1,
            )
            results[source_state] = {
                "before": before,
                "after": after,
                "solved": _expected_state(after, ("pass", "pass", "pass")),
            }
    return results


def _healthy_controls() -> dict[str, object]:
    scenarios = {
        "HC-001": [
            {
                "fingerprint": "job-10",
                "action": "poll",
                "gain": [],
                "evidence": ["10%"],
            },
            {
                "fingerprint": "job-60",
                "action": "poll",
                "gain": ["work"],
                "evidence": ["60%"],
            },
            {
                "fingerprint": "job-done",
                "action": "poll",
                "gain": ["done"],
                "evidence": ["complete"],
            },
        ],
        "HC-002": [
            {
                "fingerprint": "fail-8",
                "action": "edit-verify",
                "gain": [],
                "evidence": [],
            },
            {
                "fingerprint": "fail-5",
                "action": "edit-verify",
                "gain": ["tests"],
                "evidence": [],
            },
            {
                "fingerprint": "fail-2",
                "action": "edit-verify",
                "gain": ["tests"],
                "evidence": [],
            },
            {
                "fingerprint": "fail-0",
                "action": "edit-verify",
                "gain": ["tests"],
                "evidence": [],
            },
        ],
        "HC-003": [
            {
                "fingerprint": "world-same",
                "action": "inspect-a",
                "gain": [],
                "evidence": ["A excluded"],
            },
            {
                "fingerprint": "world-same",
                "action": "inspect-b",
                "gain": [],
                "evidence": ["B likely"],
            },
            {
                "fingerprint": "world-same",
                "action": "inspect-c",
                "gain": [],
                "evidence": ["boundary found"],
            },
        ],
        "HC-004": [
            {
                "fingerprint": "service-down",
                "action": "retry",
                "gain": [],
                "evidence": ["transient 503"],
            },
            {
                "fingerprint": "service-ready",
                "action": "retry",
                "gain": ["request"],
                "evidence": ["external recovery"],
            },
        ],
    }
    results: dict[str, object] = {}
    for name, points in scenarios.items():
        checkpoints = [
            {
                "checkpoint_id": f"{name}:cp{index}",
                "state_fingerprint": point["fingerprint"],
                "environment_facts": {"state": point["fingerprint"]},
            }
            for index, point in enumerate(points)
        ]
        progress = [
            {
                "checkpoint_id": f"{name}:cp{index}",
                "gained_requirements": point["gain"],
                "task_progress_delta": 1.0 if point["gain"] else 0.0,
                "new_evidence": point["evidence"],
                "cost_since_previous_checkpoint": {"actor_actions": 1},
            }
            for index, point in enumerate(points)
        ]
        assessment = _detect_periodic_non_progress(checkpoints, progress)
        results[name] = {
            "assessment": assessment,
            "preserved_as_healthy": assessment["verdict"] == "no_cycle",
            "points": points,
        }
    return results


def _feature_ablation_study(
    replay: Mapping[str, object],
    healthy: Mapping[str, object],
) -> dict[str, object]:
    checkpoints = replay["checkpoints"]
    progress = replay["progress"]
    recurrence = replay["recurrence"]
    assert isinstance(checkpoints, list)
    assert isinstance(progress, list)
    assert isinstance(recurrence, Mapping)
    trajectories: dict[str, dict[str, object]] = {
        "FS-001": {
            "is_failure": True,
            "actions": ["mutate-shared-policy"] * max(len(checkpoints) - 1, 0),
            "states": [item["state_fingerprint"] for item in checkpoints],
            "task_deltas": [item["task_progress_delta"] for item in progress],
            "new_evidence": [item["new_evidence"] for item in progress],
            "full_verdict": recurrence["verdict"],
        }
    }
    for name, raw in healthy.items():
        assert isinstance(raw, Mapping)
        points = raw["points"]
        assessment = raw["assessment"]
        assert isinstance(points, list)
        assert isinstance(assessment, Mapping)
        trajectories[name] = {
            "is_failure": False,
            "actions": [str(item["action"]).split("-", 1)[0] for item in points],
            "states": [item["fingerprint"] for item in points],
            "task_deltas": [1.0 if item["gain"] else 0.0 for item in points],
            "new_evidence": [item["evidence"] for item in points],
            "full_verdict": assessment["verdict"],
        }

    predictions: dict[str, dict[str, bool]] = {}
    for name, item in trajectories.items():
        actions = item["actions"]
        states = item["states"]
        task_deltas = item["task_deltas"]
        assert isinstance(actions, list)
        assert isinstance(states, list)
        assert isinstance(task_deltas, list)
        repeated_action = len(actions) != len(set(actions))
        repeated_state = len(states) != len(set(states))
        no_positive_task_progress = not any(float(value) > 0 for value in task_deltas)
        predictions[name] = {
            "repeated_action_only": repeated_action,
            "repeated_state_only": repeated_state,
            "zero_task_progress_only": no_positive_task_progress,
            "repeated_state_and_zero_task_progress": (
                repeated_state and no_positive_task_progress
            ),
            "state_progress_evidence_cost_oracle": (
                item["full_verdict"] == "non_progress_cycle"
            ),
        }

    rule_names = next(iter(predictions.values())).keys()
    rule_metrics: dict[str, object] = {}
    for rule in rule_names:
        true_positives = sum(
            prediction[rule] and trajectories[name]["is_failure"]
            for name, prediction in predictions.items()
        )
        false_positives = sum(
            prediction[rule] and not trajectories[name]["is_failure"]
            for name, prediction in predictions.items()
        )
        false_negatives = sum(
            not prediction[rule] and trajectories[name]["is_failure"]
            for name, prediction in predictions.items()
        )
        rule_metrics[rule] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
    full_metrics = rule_metrics["state_progress_evidence_cost_oracle"]
    assert isinstance(full_metrics, Mapping)
    return {
        "predictions": predictions,
        "rule_metrics": rule_metrics,
        "requirement_vector_observation": (
            "S0 and S1 both have scalar coverage 0.5; the (R1,R2) vector and "
            "regressed_requirements retain which requirement was destroyed."
        ),
        "passed": full_metrics["true_positives"] == 1
        and full_metrics["false_positives"] == 0
        and full_metrics["false_negatives"] == 0,
    }


async def _run_local_experiments() -> dict[str, object]:
    manifest_validation = _validate_fixture_manifest()
    baseline: dict[str, object]
    with tempfile.TemporaryDirectory(prefix="ejagent-fs001-baseline-") as tmp:
        root = _copy_baseline(Path(tmp))
        checkpoint = _checkpoint(
            root,
            trial_id="fs001-baseline",
            sequence=0,
            causal_actions=[],
            actor_action_count=0,
        )
        baseline = {
            "checkpoint": checkpoint,
            "matches_expected_s0": _expected_state(
                checkpoint, ("fail", "pass", "pass")
            ),
        }

    gold = _gold_controls()

    with tempfile.TemporaryDirectory(prefix="ejagent-fs001-replay-") as tmp:
        replay_root = _copy_baseline(Path(tmp))
        replay = await _deterministic_replay(replay_root)

    healthy = _healthy_controls()
    context_timing = await _context_timing_controls()
    feature_ablation = _feature_ablation_study(replay, healthy)
    gates = {
        "fixture_manifest_valid": bool(manifest_validation["valid"]),
        "baseline_s0": bool(baseline["matches_expected_s0"]),
        "gold_from_s0": bool(gold["S0"]["solved"]),
        "gold_from_s1": bool(gold["S1"]["solved"]),
        "replay_runtime_permitted_cycle": (
            replay["outcome"]["status"] == RunStatus.FAILED.value
            and replay["outcome"]["stop_reason"] == StopReason.MAX_STEPS.value
            and replay["recurrence"]["verdict"] == "non_progress_cycle"
        ),
        "shadow_analyzer_matches_offline_oracle": (
            replay["shadow_report"]["verdict"] == replay["recurrence"]["verdict"]
            and replay["shadow_report"]["period"] == replay["recurrence"]["period"]
            and replay["shadow_report"]["candidate_checkpoint_ids"]
            == replay["recurrence"]["candidate_checkpoint_ids"]
        ),
        "all_healthy_controls_preserved": all(
            bool(item["preserved_as_healthy"]) for item in healthy.values()
        ),
        "context_event_timing": bool(context_timing["passed"]),
        "feature_ablation_separates_controls": bool(feature_ablation["passed"]),
    }
    return {
        "experiment": "trajectory-preimplementation-local-v1",
        "fixture_manifest_validation": manifest_validation,
        "baseline": baseline,
        "gold_controls": gold,
        "deterministic_replay": replay,
        "context_timing_controls": context_timing,
        "feature_ablation_study": feature_ablation,
        "healthy_controls": healthy,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def _live_preregistration(config: ModelConfig) -> dict[str, object]:
    preregistration = json.loads(LIVE_PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    expected = {
        "model": preregistration["model"],
        "temperature": preregistration["generation"]["temperature"],
        "timeout": preregistration["generation"]["timeout_seconds_per_request"],
        "include_usage": preregistration["generation"]["include_usage"],
    }
    actual = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "include_usage": config.include_usage,
    }
    if actual != expected:
        raise RuntimeError(
            "Provider settings do not match the frozen FS-001 preregistration: "
            f"expected={expected!r}, actual={actual!r}"
        )
    return preregistration


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _audit_records(outcome: object) -> list[dict[str, object]]:
    records = getattr(outcome, "audit_records", ())
    return [
        {
            "run_id": record.run_id,
            "sequence": record.sequence,
            "kind": record.kind,
            "occurred_at": record.occurred_at.isoformat(),
            "payload": _json_safe(record.payload),
        }
        for record in records
    ]


def _attribute_action_batches(
    state: _FixtureToolState,
    outcome: object,
) -> bool:
    by_call_id = {str(item["tool_call_id"]): item for item in state.actions}
    causally_ambiguous = False
    for record in getattr(outcome, "audit_records", ()):
        if record.kind != "assistant_message":
            continue
        turn = int(record.payload["turn"])
        calls = record.payload["tool_calls"]
        mutation_count = sum(call["name"] == "write_file" for call in calls)
        causally_ambiguous = causally_ambiguous or mutation_count > 1
        for call in calls:
            action = by_call_id.get(str(call["id"]))
            if action is None:
                continue
            action["model_turn"] = turn
            action["batch_id"] = f"{state.trial_id}:turn:{turn}"
            action["proposed_at"] = record.occurred_at.isoformat()
    return causally_ambiguous


def _failure_record(outcome: object) -> dict[str, object] | None:
    failure = getattr(outcome, "failure", None)
    if failure is None:
        return None
    return {
        "phase": failure.phase.value,
        "code": failure.code.value,
        "message": failure.message,
        "retryable": failure.retryable,
    }


def _classify_live_trial(
    *,
    final_checkpoint: Mapping[str, object],
    recurrence: Mapping[str, object],
    outcome: object | None,
    protocol_error: str | None,
    timed_out: bool,
    causally_ambiguous: bool,
) -> str:
    if _expected_state(final_checkpoint, ("pass", "pass", "pass")):
        return "solved"
    if timed_out:
        return "infrastructure_failure"
    if protocol_error is not None:
        return "protocol_failure"
    if outcome is None:
        return "infrastructure_failure"
    failure = getattr(outcome, "failure", None)
    if failure is not None and failure.code.value in {
        "provider_error",
        "rate_limit",
        "timeout",
        "authentication",
        "tool_error",
    }:
        return "infrastructure_failure"
    if failure is not None and failure.code.value == "runtime_error":
        return "protocol_failure"
    if causally_ambiguous:
        return "causally_ambiguous"
    result = outcome.result
    if (
        recurrence["verdict"] == "non_progress_cycle"
        and result.stop_reason is not StopReason.TEXT_RESPONSE
    ):
        return "non_progress_cycle"
    if result.status is RunStatus.COMPLETED:
        return "premature_completion"
    if result.stop_reason is StopReason.MAX_STEPS:
        return "max_steps_without_cycle"
    if result.status is RunStatus.CANCELLED:
        return "cancelled"
    return "infrastructure_failure"


async def _run_live_trial(
    *,
    config: ModelConfig,
    preregistration: Mapping[str, object],
    condition: str,
    trial_number: int,
) -> dict[str, object]:
    protocol_version = str(preregistration["protocol_version"])
    trial_id = f"{protocol_version}:{condition}:{trial_number:02d}"
    run_limits = preregistration["run_limits"]
    assert isinstance(run_limits, Mapping)
    started_at = _utc_timestamp()
    started_monotonic = time.monotonic()
    outcome = None
    protocol_error: str | None = None
    timed_out = False

    with tempfile.TemporaryDirectory(
        prefix=f"ejagent-{condition}-{trial_number}-"
    ) as tmp:
        root = _copy_baseline(Path(tmp))
        state = _FixtureToolState(root=root, trial_id=trial_id, checkpoints=[])
        state.checkpoints.append(
            _checkpoint(
                root,
                trial_id=trial_id,
                sequence=0,
                causal_actions=[],
                actor_action_count=0,
            )
        )
        context = (
            None
            if condition == "baseline_identity"
            else _ExperimentContextPipeline(state, condition)
        )
        model = OpenAIModelPort(config)
        kernel = RuntimeKernel(
            model=model,
            tools=_live_fixture_tools(state),
            context=context,
        )
        try:
            await model.start()
            outcome = await asyncio.wait_for(
                kernel.run(
                    RunSpec(
                        run_id=trial_id,
                        base_revision=0,
                        intent=RunIntent.TASK,
                        task=str(preregistration["task_message"]),
                        messages=(
                            SystemMessage(str(preregistration["system_message"])),
                        ),
                        limits=RunLimits(
                            max_turns=int(run_limits["max_turns"]),
                            max_repeated_tool_calls=int(
                                run_limits["max_repeated_tool_calls"]
                            ),
                        ),
                        configuration_revision=protocol_version,
                        metadata={"context_condition": condition},
                    )
                ),
                timeout=float(run_limits["trial_timeout_seconds"]),
            )
        except TimeoutError:
            timed_out = True
        except (
            Exception
        ) as exc:  # experiment retains protocol failures verbatim by type
            protocol_error = f"{type(exc).__name__}: {exc}"
        finally:
            await model.shutdown()

        final_checkpoint = _checkpoint(
            root,
            trial_id=trial_id,
            sequence=len(state.checkpoints),
            causal_actions=[],
            actor_action_count=state.actor_action_count,
        )
        progress = _progress_records(state.checkpoints)
        recurrence = _detect_periodic_non_progress(state.checkpoints, progress)
        causally_ambiguous = (
            _attribute_action_batches(state, outcome) if outcome is not None else False
        )
        classification = _classify_live_trial(
            final_checkpoint=final_checkpoint,
            recurrence=recurrence,
            outcome=outcome,
            protocol_error=protocol_error,
            timed_out=timed_out,
            causally_ambiguous=causally_ambiguous,
        )
        terminal_result = None
        raw_audit: list[dict[str, object]] = []
        failure = None
        if outcome is not None:
            terminal_result = {
                "run_id": outcome.result.run_id,
                "status": outcome.result.status.value,
                "stop_reason": outcome.result.stop_reason.value,
                "turns": outcome.result.turns,
                "output": outcome.result.output,
                "usage": outcome.result.usage.to_dict(),
            }
            raw_audit = _audit_records(outcome)
            failure = _failure_record(outcome)

        ended_at = _utc_timestamp()
        return {
            "trial_id": trial_id,
            "fixture_version": "fs001-v1",
            "provider_and_model": {
                "provider_family": "openai-compatible",
                "endpoint_sha256": hashlib.sha256(config.base_url.encode()).hexdigest(),
                "model": config.model,
            },
            "generation_configuration": {
                "temperature": config.temperature,
                "timeout_seconds_per_request": config.timeout,
                "include_usage": config.include_usage,
            },
            "context_condition": condition,
            "run_limits": dict(run_limits),
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
            "run_id": trial_id,
            "agent_id": f"fs001-agent-{condition}-{trial_number:02d}",
            "terminal_result": terminal_result,
            "failure": failure,
            "protocol_error": protocol_error,
            "timed_out": timed_out,
            "primary_classification": classification,
            "causally_ambiguous": causally_ambiguous,
            "actions": state.actions,
            "observations": state.observations,
            "checkpoints": state.checkpoints,
            "progress": progress,
            "recurrence": recurrence,
            "completion_audit": final_checkpoint,
            "run_audit": raw_audit,
        }


def _range(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"median": None, "min": None, "max": None}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _live_aggregate(trials: list[dict[str, object]]) -> dict[str, object]:
    classifications: dict[str, int] = {}
    for trial in trials:
        classification = str(trial["primary_classification"])
        classifications[classification] = classifications.get(classification, 0) + 1
    valid = [
        trial
        for trial in trials
        if trial["primary_classification"]
        not in {"infrastructure_failure", "protocol_failure"}
    ]
    valid_count = len(valid)
    turns = [
        int(trial["terminal_result"]["turns"])
        for trial in trials
        if isinstance(trial["terminal_result"], Mapping)
    ]
    tool_calls = [len(trial["actions"]) for trial in trials]
    total_tokens = [
        int(trial["terminal_result"]["usage"]["total_tokens"])
        for trial in trials
        if isinstance(trial["terminal_result"], Mapping)
    ]
    return {
        "trial_count": len(trials),
        "valid_trial_count": valid_count,
        "classification_counts": classifications,
        "cycle_incidence": (
            classifications.get("non_progress_cycle", 0) / valid_count
            if valid_count
            else None
        ),
        "solve_rate": (
            classifications.get("solved", 0) / valid_count if valid_count else None
        ),
        "premature_completion_rate": (
            classifications.get("premature_completion", 0) / valid_count
            if valid_count
            else None
        ),
        "ambiguous_trial_count": classifications.get("causally_ambiguous", 0),
        "turns": _range(turns),
        "tool_calls": _range(tool_calls),
        "total_tokens": _range(total_tokens),
        "elapsed_seconds": _range(
            [float(trial["elapsed_seconds"]) for trial in trials]
        ),
        "checkpoints": _range([len(trial["checkpoints"]) for trial in trials]),
    }


async def _run_live_experiments() -> dict[str, object]:
    config = ModelConfig.from_env()
    preregistration = _live_preregistration(config)
    conditions = preregistration["context_conditions"]
    assert isinstance(conditions, list)
    trials_per_condition = int(preregistration["trials_per_condition"])
    trials: list[dict[str, object]] = []
    for condition in conditions:
        for trial_number in range(1, trials_per_condition + 1):
            trial = await _run_live_trial(
                config=config,
                preregistration=preregistration,
                condition=str(condition),
                trial_number=trial_number,
            )
            trials.append(trial)
            print(
                f"live_trial={trial['trial_id']} "
                f"classification={trial['primary_classification']}",
                file=sys.stderr,
                flush=True,
            )
    by_condition = {
        str(condition): _live_aggregate(
            [trial for trial in trials if trial["context_condition"] == condition]
        )
        for condition in conditions
    }
    aggregate = _live_aggregate(trials)
    invalid_count = aggregate["classification_counts"].get(
        "infrastructure_failure", 0
    ) + aggregate["classification_counts"].get("protocol_failure", 0)
    return {
        "experiment": "trajectory-preimplementation-live-v1",
        "preregistration": preregistration,
        "preregistration_sha256": _file_digest(LIVE_PREREGISTRATION_PATH),
        "trials": trials,
        "aggregate": aggregate,
        "by_condition": by_condition,
        "experiment_valid": len(trials) == len(conditions) * trials_per_condition
        and invalid_count == 0,
    }


def _summary(report: Mapping[str, object]) -> dict[str, object]:
    replay = report["deterministic_replay"]
    assert isinstance(replay, Mapping)
    return {
        "experiment": report["experiment"],
        "all_gates_passed": report["all_gates_passed"],
        "gates": report["gates"],
        "replay_outcome": replay["outcome"],
        "recurrence": replay["recurrence"],
    }


def _live_summary(report: Mapping[str, object]) -> dict[str, object]:
    return {
        "experiment": report["experiment"],
        "experiment_valid": report["experiment_valid"],
        "aggregate": report["aggregate"],
        "by_condition": report["by_condition"],
        "trial_classifications": {
            trial["trial_id"]: trial["primary_classification"]
            for trial in report["trials"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write the complete local report to this generated artifact path.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run every frozen real-Provider trial (may incur usage cost).",
    )
    parser.add_argument(
        "--live-json-output",
        type=Path,
        help="Write the complete live-Provider report to this artifact path.",
    )
    args = parser.parse_args()
    report = asyncio.run(_run_local_experiments())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    output: dict[str, object] = {"local": _summary(report)}
    success = bool(report["all_gates_passed"])
    if args.live:
        live_report = asyncio.run(_run_live_experiments())
        if args.live_json_output is not None:
            args.live_json_output.parent.mkdir(parents=True, exist_ok=True)
            args.live_json_output.write_text(
                json.dumps(live_report, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        output["live"] = _live_summary(live_report)
        success = success and bool(live_report["experiment_valid"])
    elif args.live_json_output is not None:
        parser.error("--live-json-output requires --live")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
