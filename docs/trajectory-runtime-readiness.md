# Harness Trajectory Integration and Readiness

## Decision

Online trajectory observation and Context feedback are implemented Harness
capabilities. `AgentHarness` accepts an optional monitor, its `RuntimeKernel`
supplies the execution boundaries, and the host composes evaluation with
Context projection. Observation remains the default. The separate, opt-in
`CompletionPolicy` now supports bounded same-Run completion retries; see the
[current evaluation guide](evaluation.md).

The original observation-phase executable gate is
[`runtime_readiness.py`](../experiments/trajectory/runtime_readiness.py). Its
durable result is
[`2026-09-03-runtime-readiness.json`](../experiments/trajectory/results/2026-09-03-runtime-readiness.json):
1,674 bytes, SHA-256
`376b9fe31ef7ceea66b596e8800e44753d050a6fed42844248660dd7b2a4d8b6`.
All 23 gates pass.

## What is now implemented

`TrajectoryMonitor` in `ejagent.kernel.trajectory` is the stable Kernel-facing
seam; `OnlineTrajectoryMonitor` is its internal implementation. It:

- serializes capture independently for each Run;
- assigns ordered Checkpoint identities;
- validates turn, cumulative cost, and complete Action-batch attribution;
- calls a host-owned `CheckpointEvaluator` for current environment truth;
- produces an online `TrajectoryAssessment` without requiring terminal Audit;
- maps the Assessment to the next Context event;
- exposes `completion_allowed` for a Completion Claim;
- optionally stages an exact-next-turn Context frame through an update sink;
- returns immutable history and releases per-Run state through `close_run`; an
  optional close sink can archive that history and clear staged Context.

The public `ejagent.evaluation` module supplies a reusable evaluator, optional
model judge, and adapters. Hosts supply criteria and domain evidence. Core does not
infer Facts from model narration or assume that every environment is a source
repository.

## Checkpoint policy

The observation protocol supports these semantic boundaries. The Kernel emits
baseline, Tool-batch completion, and text-completion signals automatically when
a monitor is supplied; verification and external-change signals require host
integration.

| Trigger | Execution or host boundary | Causal rule |
| --- | --- | --- |
| `baseline` | after Run-local initialization, before turn 1 Context | no Actor Actions; turn 0 |
| `tool_batch_completed` | after the full concurrent batch has completed and all results have entered the workspace in proposal order | one named batch containing every proposed Action |
| `verification_completed` | after a host verifier publishes authoritative Evidence | no unrecorded Actor transition |
| `external_change` | after an external source reports a change | replaced Facts must be explicitly invalidated |
| `completion_proposed` | after a text Completion Claim or a planned Tool completion, before accepting terminal success | full Requirement/Constraint evaluation |

Audit deltas, streamed tokens, arbitrary timestamps, and individual members of
a concurrent Tool batch are not Checkpoint boundaries.

## Harness composition and Kernel wiring

The wiring is opt-in and preserves current behavior when no monitor is
supplied. It has four insertion points in `RuntimeKernel.run`:

1. Accept an optional `TrajectoryMonitor` dependency and capture `baseline`
   after `_RunWorkspace`, usage, Audit, and Tool definitions exist, before the
   first `_build_context` call.
2. After `_execute_tools` returns and every result is appended to the workspace,
   capture one `tool_batch_completed` signal. Never capture from individual
   concurrent task-completion callbacks.
3. When an assistant message contains no Tool calls, capture
   `completion_proposed` before `_terminal`. `completion_allowed` is recorded
   as advice in observation mode. Enforcement rejects an unapproved completion
   and stages feedback for the next turn while budget remains.
4. Close the monitor's Run state on every terminal, failure, cancellation, and
   protocol-exception path. A failed Completion Audit can continue the same Run
   under the policy recorded in ADR 0001.

The Kernel signal's cumulative cost comes from the existing usage accumulator,
the count of model-proposed Actions, and a monotonic Run-local clock. Causal
Action signatures contain the Tool name and a canonical-argument SHA-256
digest, not raw arguments. The Context pipeline remains the only model-facing seam: a
host composes the monitor's update sink with `TrajectoryContextBuffer` and
`TrajectoryContextPipeline`.

Capture success produces a `trajectory_checkpointed` Audit record. Capture
failure produces `trajectory_capture_failed`, disables later captures in the
same Run, and leaves the existing Run result unchanged in observation mode.
Enforcement cannot approve a completion after capture failure. The monitor is closed
from `finally`, including when a Kernel protocol error escapes. See
[ADR 0002](adr/0002-runtime-owns-trajectory-observation-boundary.md).

The [Streamlit Harness example](../examples/streamlit_runtime.py) composes a
formal `GoalEvaluator`, `EvaluationMonitor`, and Context pipeline. Feedback is enabled by
default in that example, while the library remains opt-in. It assesses recorded
probe completion and overlap, exposes checkpoints and built Context
instructions, and cleans up Run-local evaluation and buffer state. Its recovery
demo changes the scripted Actor's actions in response to a confirmed-cycle
instruction; the monitor does not directly command Tool execution.

Monitor capture failures are isolated as described above. Context projection
has its own validation: incomplete or stale Facts can raise
`ContextProtocolError` in the strict full-fact path. The formal evaluator uses a
separate limited projection for unavailable evidence. Enabling observation alone
does not enable feedback,
and observation's failure isolation does not override Context error semantics.

## Automated evidence

The readiness gate proves:

- CP1 suspicion and CP2 confirmation for a period-one online replay;
- no terminal `RunAudit` is needed for the online Assessment;
- a Requirement gain plus new Constraint violation is Regression, not Task
  Progress;
- Checkpoint cost deltas include Actions, requests, tokens, and elapsed time;
- every declared trigger is captured in order;
- concurrent causal ambiguity fails closed;
- external change requires explicit Fact invalidation;
- failed Completion Audit can produce a same-Run continuation instruction;
  the Kernel still accepts a terminal text response in this integration;
- events are staged only for their next Decision Boundary;
- prior Phase-2 failure and healthy-control gates remain green;
- `RuntimeKernel` and stable `ejagent.contracts` do not import the internal
  trajectory package; the internal implementation depends on the Kernel-owned
  seam.

## Boundary of this decision

This integration does not justify a default-on controller, automatic Action
denial, cycle termination, or top-level public contracts. Those require
post-integration telemetry and separate policy evidence.
