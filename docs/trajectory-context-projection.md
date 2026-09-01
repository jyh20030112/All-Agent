# Trajectory Context Projection

## Status

Phase 2 provides an internal, opt-in Context projection under
`ejagent._trajectory`. It is not exported from the public package, is disabled
unless a host explicitly supplies a `TrajectoryContextPipeline`, and gives no
trajectory code authority to admit Actions or terminate Runs.

The entry gates are executable in
[`phase2_evidence.py`](../experiments/trajectory/phase2_evidence.py), with the
durable result in
[`2026-09-01-phase2-summary.json`](../experiments/trajectory/results/2026-09-01-phase2-summary.json).
The reproduced JSON is 3,271 bytes with SHA-256
`7d75fdbd5f7c08740186de6909170c27cd1ff9698f28efb1ef36b2ecc9964d05`.

## Module and Interface

`TrajectoryContextProjector` is the pure deep module:

```python
projection = projector.project(frame)
```

The `TrajectoryContextFrame` joins four evaluator-owned inputs for one
Decision Boundary:

- the stable Goal anchor;
- one explicit `TrajectoryCheckpoint`;
- its `ProgressSnapshot`;
- one `TrajectoryContextEvent` explaining why the next decision should—or
  should not—receive additional Context.

`TrajectoryContextPipeline` is the small adapter at the existing
`ContextPipeline` seam. It builds the host's base Context first, obtains a
frame for the current `run_id` and `turn`, and appends at most one disposable
`TransientInstruction`:

```text
RuntimeKernel._build_context
  -> host-selected base ContextPipeline
  -> host TrajectoryContextSource(ContextRequest)
  -> TrajectoryContextProjector.project(frame)
  -> optional TransientInstruction
  -> next model request
```

RuntimeKernel depends only on the stable `ContextPipeline` Interface and has no
dependency on trajectory detection.

## Fact validity

An `EnvironmentFact` is immutable. Its checkpoint-relative `FactValidity` is
one of:

| Validity | Meaning | Model projection |
| --- | --- | --- |
| `current` | Freshness condition still holds | May appear as current truth |
| `invalidated` | A named checkpoint or event made it historical | May appear only in an explicit invalidation delta |
| `stale` | Its freshness condition no longer holds | Projection fails closed |
| `unknown` | Freshness cannot currently be established | Projection fails closed |

A current Fact carries its subject, predicate, frozen value, claim scope,
source, observation time, checkpoint, Evidence reference, freshness condition,
and authority. Complete capture is required for model-facing projection.
State fingerprints remain controller-only.

## Event visibility

The default policy is deliberately asymmetric:

| Event | Visible to next model call | Projected consequence |
| --- | --- | --- |
| `FactsUpdated` | yes | current Facts and scoped Evidence |
| `ProgressEvaluated` | yes | Task/Epistemic Progress and Regression |
| `CycleSuspected` | no | gather stronger Evidence first |
| `CycleConfirmed` | yes | Goal anchor, current Facts, exhausted Action path, replan request |
| `ConstraintViolated` | yes | violated item and required recovery boundary |
| `ExternalStateChanged` | yes | invalidated Facts and refreshed current State |
| `CompletionAuditFailed` | yes | unmet items, missing Evidence, continue-current-Run instruction |

Events are exposed at the next Decision Boundary, not when they are merely
recorded. Tests run a real two-turn `RuntimeKernel` and verify that a frame for
turn 2 is absent from turn 1 and visible exactly once on turn 2.

## Phase-2 entry gates

| Gate | Evidence |
| --- | --- |
| More than FS-001 | deployment-routing supplies a second confirmed period-two failure; six additional domains exercise controls |
| Provenance/freshness/invalidation | typed Facts are required by projection; stale or unknown validity fails closed |
| Concurrent causal attribution | unattributed batches return `causally_ambiguous`; complete batch paths may be assessed |
| False-positive review | productive wait, exploration, legitimate retry, and regress-then-recover all remain `no_cycle` |
| Completion Audit Run semantics | [ADR 0001](adr/0001-failed-completion-audit-continues-run.md) chooses same-Run feedback while budget remains |

## Current non-goals

- no default or public Context pipeline;
- no universal Fact collector or durable Fact store;
- no automatic event producer inside Runtime;
- no Completion Audit implementation yet;
- no Action denial, cancellation, or termination policy;
- no projection of detector thresholds, fingerprints, stale values, or the full
  trajectory log.

The host still owns Fact capture and event production. Phase 2 establishes the
model-facing seam and its invariants without claiming those domain-specific
responsibilities for Core.
