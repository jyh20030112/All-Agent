# Harness Trajectory Context Feedback

## Status

The Harness supports internal, opt-in trajectory Context projection under
`ejagent._trajectory`. A host composes `AgentHarness` with a monitor and
`TrajectoryContextPipeline` to provide decision-specific feedback. The
projection is not a stable top-level API and gives no trajectory code authority
to admit Actions or terminate Runs. The Streamlit example enables this
composition by default; the library does not.

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

RuntimeKernel depends on the stable `ContextPipeline` and optional
`TrajectoryMonitor` Interfaces, and has no dependency on trajectory detection
or Fact-model internals.

`OnlineTrajectoryMonitor` now produces a `TrajectoryUpdate` immediately after
each captured semantic boundary. An optional synchronous update sink can stage
`update.to_context_frame(...)` in `TrajectoryContextBuffer`; the buffer keys
frames by exact `(run_id, turn)`. This composition connects online assessment
to the existing pipeline without making Conversation an event store or
exposing the current event before the next model decision.

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

## Current projection boundaries

- no default-on trajectory pipeline in the library or stable top-level export;
- no universal Fact collector or durable Fact store;
- no default-on capture or enforcement inside `RuntimeKernel`;
- no domain-independent Completion verifier; the host evaluator supplies its
  authoritative Requirement and Constraint verdicts;
- no Action denial, cancellation, or termination policy;
- no projection of detector thresholds, fingerprints, stale values, or the full
  trajectory log.

The host owns domain Fact capture and evaluation. `OnlineTrajectoryMonitor`
generates events from those evaluations and analysis; the host connects its
update sink to the Context source. The projection is a Harness feedback
capability whose correctness depends on these domain inputs.

The internal capture/event/context seam has passed the
[Harness integration gates](trajectory-runtime-readiness.md) and is now wired into
`RuntimeKernel` when a host explicitly supplies a monitor. This does not enable
enforcement by itself.

`TrajectoryContextBuffer` keys frames by Run and turn. Reads do not consume a
frame, so rebuilding Context for the same turn returns the same staged input;
`close_run()` removes its frames. Terminal completion advice may be staged for
a next turn that never occurs, because the current Kernel does not enforce
completion review. Applications should distinguish assessments from instructions
actually included in a model Context, as the Streamlit example does.
