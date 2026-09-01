# Trajectory Shadow Analysis

## Status

The first observation-only implementation is present under
`ejagent._trajectory`. It is intentionally internal, is not re-exported by the
top-level package, and does not change Runtime execution, completion, Context,
or Tool admission.

The implementation follows the terminology in [the domain glossary](../CONTEXT.md)
and the evidence established by the
[pre-implementation experiment](trajectory-experiment-report.md).

## Module and Interface

`ShadowTrajectoryAnalyzer` is the deep module. Its external Interface is one
pure operation:

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
acquiring Runtime control authority.

## Existing seam

`ShadowTrajectoryObserver` is a small adapter at the existing `RunObserver`
seam:

```text
Runtime finishes
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

### Progress

Requirement coverage is derived from the vector but does not replace it.
Every snapshot retains Requirements gained and regressed independently. An
empty Constraint set is valid; an empty Requirement set is not.

`new_evidence` is supplied by the host evaluator. Audit activity or model
narration is never promoted to Epistemic Progress automatically.

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

## Current non-goals

- no additions to stable `ejagent.contracts`;
- no top-level public export;
- no in-Run checkpoint hook;
- no Context projection;
- no Completion Audit enforcement;
- no Action denial, replan, cancellation, or Run termination;
- no universal State Fingerprint or Fact persistence schema.

## Entry criteria for the next phase

An opt-in model-facing Context projection should not be implemented until:

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

Until those criteria are met, `TrajectoryReport` is measurement output rather
than Controller policy.

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
