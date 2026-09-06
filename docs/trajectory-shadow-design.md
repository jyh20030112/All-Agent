# Trajectory Shadow Analysis

## Status

Trajectory assessment is an internal Harness capability under
`ejagent._trajectory`. The analyzer provides pure online assessment and
post-Run reports; by itself it changes neither execution nor Context. The
online monitor and Context adapter compose these assessments into feedback,
as described in [Harness trajectory integration](trajectory-runtime-readiness.md).
The package is not re-exported by the top-level API and has no enforcement
authority over Tool admission or Run completion.

The implementation follows the terminology in [the domain glossary](../CONTEXT.md)
and the evidence established by the
[pre-implementation experiment](trajectory-experiment-report.md).

## Module and Interface

`ShadowTrajectoryAnalyzer` is the deep assessment module. It now has a pure
online operation that does not require a terminal Run:

```python
assessment = analyzer.assess(environment_checkpoints)
```

The terminal Shadow adapter enriches the same Assessment with normalized
Audit Actions and Observations:

```python
report = analyzer.analyze(run_audit, environment_checkpoints)
```

The caller supplies two independently authoritative inputs:

- `RunAudit`: the current Core record of model, Tool, control, and terminal
  events;
- `TrajectoryCheckpoint` values: host-owned Environment Facts, Requirement and
  Constraint verdicts, Evidence deltas, causal Action signatures, and cost.

The module returns one immutable `TrajectoryReport` containing:

- model proposal-order Actions normalized from Audit;
- Tool Observations in actual completion order;
- terminal status, Store commit decision, revisions, turns, requests, and token
  cost;
- per-checkpoint Progress and Regression;
- `no_cycle`, `cycle_suspected`, `non_progress_cycle`,
  `causally_ambiguous`, or `insufficient_evidence`;
- the supporting checkpoint window and diagnostics.

Fact capture, persistence, model prompting, and Action admission are hidden
from neither the caller nor the module: they are explicitly outside this
Interface. This prevents the Analyzer from inventing environment truth or
acquiring Kernel control authority.

## Existing seam

`ShadowTrajectoryObserver` is a small adapter at the existing `RunObserver`
seam:

```text
Kernel finishes
  -> Store commit decision
  -> immutable RunAudit
  -> ShadowTrajectoryObserver
       -> host checkpoint source(run_id)
       -> ShadowTrajectoryAnalyzer.analyze(...)
       -> host report sink(report)
```

The Harness dispatches observers after the Store decision and ignores observer
failures with respect to the completed Run. Therefore this adapter can record
and measure trajectory behavior but cannot alter the Run it observes.

The checkpoint source and report sink remain host-owned. EJAgent does not yet
claim a universal environment adapter or persistence schema.

## Assessment invariants

### Environment equivalence

Fingerprint equality only creates a candidate. Equivalent checkpoints must
also have equal:

- projection version;
- underlying Environment Facts;
- Requirement vector;
- Constraint vector.

Causally incomplete checkpoints cannot prove recurrence.

### Progress and cost

Requirement coverage is derived from the vector but does not replace it.
Every snapshot retains Requirements gained and regressed independently. An
empty Constraint set is valid; an empty Requirement set is not.

`requirement_coverage_delta` is the raw scalar coverage change.
`task_progress_delta` is only numeric while every Constraint is satisfied; it
is `None` while the Goal is blocked by a false or unresolved Constraint.
`ProgressStatus` supplies the task-level interpretation. A newly violated
Constraint or lost Requirement is `regressed`, even if raw Requirement
coverage increased in the same transition.

`new_evidence` is supplied by the host evaluator. Audit activity or model
narration is never promoted to Epistemic Progress automatically.

Each Checkpoint may carry cumulative Actor Action, model request, token, and
elapsed-time cost. The Progress Snapshot derives the non-negative difference
from the previous Checkpoint and fails closed on regressing counters.

### Cycle thresholds

For period two:

```text
CP0 S0 -A-> CP1 S1 -B-> CP2 S0 -A-> CP3 S1
  => cycle_suspected

CP0 S0 -A-> CP1 S1 -B-> CP2 S0 -A-> CP3 S1 -B-> CP4 S0
  => non_progress_cycle, if the remaining oracle conditions hold
```

Confirmation requires:

- two complete equivalent State cycles;
- the complete semantic Action path repeated;
- equivalent underlying Facts, not only fingerprints;
- no sustained Task Progress or new best coverage in the repeated window;
- no new Evidence in the repeated window;
- increasing Actor Action cost;
- causally complete checkpoints.

## Test surface

Tests cross the same `analyze(audit, checkpoints)` Interface used by the
experiment adapter. They cover:

- CP3 suspicion and CP4 confirmation;
- mismatched underlying Facts despite equal fingerprints;
- incomplete causal attribution;
- goals without Constraints;
- FS-001 and HC-001 through HC-004;
- concurrent Tool proposal order versus completion order;
- actual Harness after-commit Observer dispatch;
- differential agreement with the frozen experiment oracle;
- replay of all nine pre-registered live trial Audits.

The same suite also crosses `assess(checkpoints)` directly and proves that its
cycle result agrees with the terminal report, so Kernel integration does not
need to manufacture a partial `RunAudit`.

## Analyzer boundaries

- no additions to stable `ejagent.contracts`;
- no top-level public export;
- no implicit observation or Context wiring in the library; applications
  explicitly compose the monitor and projection;
- no Completion Audit enforcement;
- no direct Action denial, forced replanning, cancellation, or Run termination;
  a Context adapter can ask the Actor to replan;
- no universal State Fingerprint or Fact persistence schema.

## Historical Phase-2 entry criteria

The initial study required the following evidence before Context projection:

1. Shadow reports have been collected across more than the single FS-001
   domain.
2. Host checkpoint sources demonstrate explicit Fact provenance, freshness,
   and invalidation.
3. Concurrent mutation batches are either causally attributed or excluded from
   confirmation.
4. False-positive review includes productive waits, exploration, retries, and
   multi-step regress-then-recover work.
5. The project decides whether a failed Completion Audit continues the current
   Run or starts a new Run.

These criteria were evaluated in Phase 2 below. Passing them enabled feedback
composition; `TrajectoryReport` remains assessment output rather than an
execution policy.

## Phase-2 resolution

The criteria above are now represented as executable gates in
[`phase2_evidence.py`](../experiments/trajectory/phase2_evidence.py). They pass
for a second failure domain, four named false-positive controls, explicit Fact
validity, and both attributed and excluded concurrent batches. Completion Audit
failure semantics are recorded in
[ADR 0001](adr/0001-failed-completion-audit-continues-run.md).

The resulting opt-in model-facing module is specified in
[`trajectory-context-projection.md`](trajectory-context-projection.md). Shadow
reports remain measurement output; the new Context adapter consumes a separate
host-owned frame and does not give the after-commit Observer in-Run authority.

## Online integration

`OnlineTrajectoryMonitor` serializes Checkpoint capture per Run, delegates
fresh truth to a host `CheckpointEvaluator`, assesses the result immediately,
maps it to a Context event, and exposes explicit lifecycle cleanup. The first
opt-in Kernel wiring now crosses a stable, Kernel-owned observation seam; the
readiness evidence and implemented insertion points are documented in
[`trajectory-runtime-readiness.md`](trajectory-runtime-readiness.md).
