# Trajectory Pre-implementation Experiment Report

## Outcome

The pre-implementation experiments reproduced a solvable non-progress cycle in
the current Runtime, preserved all four healthy controls, validated the
proposed context-event timing, and completed nine pre-registered live-model
trials. No production Runtime or contract code was changed.

The central result is deliberately narrower than “models loop”: the current
Runtime permits a period-two environment-State cycle that its consecutive
equal-Tool-call guard cannot recognize. A deterministic policy reproduced that
failure. The sampled live model solved FS-001 in all nine trials and therefore
did not estimate a non-zero natural incidence for this fixture.

The durable compact result is
[`../experiments/trajectory/results/2026-09-01-summary.json`](../experiments/trajectory/results/2026-09-01-summary.json).

## Frozen experiment

- Fixture: `fs001-v1`, frozen clock, manifest-verified source and verifiers.
- Goal: reject expired access tokens on the protected path while preserving a
  valid refresh flow.
- Requirements: R1 protected-path rejection; R2 refresh preservation.
- Constraint: C1 public signatures and token payload compatibility.
- Completion Audit: R1, R2, and C1 must pass at the same checkpoint.
- Correct solution control: path-sensitive internal validation from both S0
  and S1.
- Failure replay: alternate the same global policy between strict and legacy.
- Healthy controls: productive polling, productive edit/verify, evidence-gain
  exploration, and legitimate retry after external recovery.

The live protocol was frozen before execution in
[`../experiments/trajectory/fs001/live-preregistration.json`](../experiments/trajectory/fs001/live-preregistration.json).

## Local results

All eight local gates passed:

| Gate | Result |
| --- | --- |
| Fixture manifest hashes and file set | pass |
| Baseline is S0 = `(R1 fail, R2 pass, C1 pass)` | pass |
| Gold solution reaches S2 from S0 | pass |
| Gold solution reaches S2 from S1 | pass |
| Current Runtime permits deterministic A/B recurrence | pass |
| HC-001 through HC-004 remain healthy | pass |
| Event-context timing follows the frozen policy | pass |
| Full oracle separates failure from all controls | pass |

The deterministic Run executed eight Tool calls over eight turns and ended as
`failed / max_steps`. The offline evaluator found a period-two recurrence:

```text
CP0 S0 -A-> CP1 S1 -B-> CP2 S0 -A-> CP3 S1 -B-> CP4 S0
```

The paired States had equal fingerprints and equal underlying Facts, no new
Evidence appeared after the first cycle, Requirement coverage remained at
`1/2`, and Action cost continued to increase. The Runtime Audit shows that the
existing repeat guard did not stop the alternating calls.

### A falsified measurement assumption

The first detector draft treated “one Requirement gained” as Task Progress.
The replay disproved that assumption: every S0/S1 transition gains one
Requirement and regresses the other, leaving coverage unchanged. The corrected
record retains both fields:

```text
task_progress_delta = 0
gained_requirements = [R1 or R2]
regressed_requirements = [R2 or R1]
```

This is why a scalar pass count cannot replace the Requirement vector.

### Feature ablation

| Candidate signal | True positives | False positives | What it misclassifies |
| --- | ---: | ---: | --- |
| Repeated Action only | 1 | 4 | every healthy control |
| Repeated State only | 1 | 1 | evidence-gaining exploration |
| Zero Task Progress only | 1 | 1 | evidence-gaining exploration |
| Repeated State + zero Task Progress | 1 | 1 | evidence-gaining exploration |
| State recurrence + no Task/Epistemic Progress + rising cost | 1 | 0 | none in this five-scenario set |

This does not prove the full oracle is universally sufficient. It establishes
that Action repetition, State repetition, or one-dimensional progress are not
sufficient even for the required controls.

## Live-model results

Provider family was OpenAI-compatible, model `glm-5`, temperature `0.7`, with
three trials per context condition. Every result was retained.

| Context condition | Valid | Solved | Median turns | Median Tools | Median tokens | Median seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current identity projection | 3 | 3 | 8 | 12 | 27,226 | 57.90 |
| Current Facts + Progress | 3 | 3 | 6 | 11 | 19,608 | 50.79 |
| Facts + Progress + confirmed-cycle intervention | 3 | 3 | 5 | 10 | 14,416 | 41.34 |

Across all nine trials:

- solve rate: `9/9`;
- observed Non-progress Cycle incidence: `0/9`;
- premature Completion: `0`;
- causal ambiguity: `0`;
- infrastructure or protocol failures: `0`;
- every Actor called V-ALL before its Completion Claim;
- every independent terminal Completion Audit passed R1, R2, and C1.

The descriptive efficiency differences are not causal evidence: each arm has
only three samples, generation is stochastic, and the confirmed-cycle arm
never actually emitted a cycle intervention because no live trajectory met the
oracle. The live result therefore says “baseline Context was sufficient for
this model in this small sample,” not “State/Progress projection has no value”
and not “trajectory cycles do not occur.”

## Context-event timing control

Because no live trajectory triggered `CycleConfirmed`, a deterministic A/B
replay validated the projection boundary independently:

| Condition | State projection visible | Cycle intervention visible |
| --- | --- | --- |
| Baseline identity | never | never |
| Facts + Progress | turns 1–8 | never |
| Confirmed-cycle condition | turns 1–8 | turn 8 |

Turn 6 is the first decision at which the repeated State window is sufficient
for `CycleSuspected`, but the second `A/B` Action path is not complete. Turn 8
is the first decision after CP4 closes `A/B/A/B`; only then is
`CycleConfirmed` projected. The model never sees the State Fingerprint; it sees
current scoped Facts, Progress/Regression, provenance, the compared checkpoint
identities, and an explicit request to replan.

## What counts as an Environment Fact

For implementation planning, an Environment Fact is:

> an immutable, source-attributed assertion about the environment at a named
> checkpoint, with declared claim scope, supporting Evidence, and an explicit
> freshness/invalidation condition.

In FS-001 that means source identity, R1/R2/C1 verdicts and failure signatures,
public signature shape, token schema, fixture version, and side-effect count.
A Tool result is only an Observation; a model statement is a Belief or
Completion Claim; “the trajectory is regressing” is an Assessment. None should
be silently promoted to Fact.

After a source mutation, previous verifier Facts become historical Evidence.
They remain auditable but are not current until V-ALL establishes a new
checkpoint. This invalidation rule is necessary because ordinary Conversation
history retains old Tool results.

## What the model should see, and when

The smallest justified policy from these experiments is:

| Decision boundary | Model-visible projection |
| --- | --- |
| Run start | stable Goal, Requirements, Constraints, Completion Evidence policy, relevant baseline Facts |
| Pure observation completes | bounded Observation or Fact Delta with provenance; no forced full State resend |
| Environment mutation completes | mark affected earlier Facts stale; after checkpointing, show current Requirement/Constraint vector and relevant State Delta |
| Verification completes | scoped verdicts, failure Evidence, Regression, current/best coverage |
| Cycle suspected with weak Facts | no warning yet; gather the missing authoritative observation |
| Cycle confirmed | Goal anchor, current Facts, equivalent checkpoint references, exhausted Action path, lack of Task/Epistemic Progress, and a replan request |
| Completion proposed | no success confirmation; run an independent Completion Audit |
| Completion Audit fails | unmet items and missing Evidence, then continue or start a new Run by policy |
| External state changes | invalidated Facts plus freshly observed replacements |

The complete trajectory remains recorder/evaluator-facing. The model receives
a bounded current projection, not a dump of Audit records, internal
fingerprints, hidden thresholds, or unrelated environment data.

## Implementation boundary supported by the evidence

The experiments support an observation-first implementation boundary:

1. Normalize existing Audit into Action and Observation records.
2. Add experiment/internal checkpoint, Fact, State, and Progress projections
   without changing public contracts first.
3. Run the evaluator offline or observationally against more real tasks.
4. Add independent Completion Audit before treating text as success.
5. Only after false-positive measurement, consider in-Run `CycleConfirmed`
   projection or Action admission policy.

They do not yet justify a blocking cycle guard, a universal fingerprint schema,
or public Core types for every conceptual term.

## Reproduction and artifacts

Local deterministic experiments:

```bash
UV_CACHE_DIR=/tmp/ejagent-uv-cache uv run python \
  experiments/trajectory/run_experiments.py \
  --json-output /tmp/ejagent-trajectory-local.json
```

Frozen live batch (uses configured Provider credentials and incurs usage):

```bash
UV_CACHE_DIR=/tmp/ejagent-uv-cache uv run python \
  experiments/trajectory/run_experiments.py \
  --live \
  --json-output /tmp/ejagent-trajectory-local.json \
  --live-json-output /tmp/ejagent-trajectory-live.json
```

Raw artifact integrity for this run:

- local: `3205fa5ee54867120396e35bc4833586307169623334f0091d637d26935db981`;
- live: `2091dd3dc40d06fa54d7110aaa45488263ebee04ae8f11eb6fb49ef419bbe891`;
- Shadow replay: `1c304a83c77556cd7e2856e6401c8fb7a182f1751f1e371b45c17e8df959dfc8`.

The raw live artifact is 3.76 MB and remains generated rather than committed.
It contains complete trial, Action, Observation, checkpoint, Progress, terminal,
and current RunAudit records. A secret scan confirmed that neither the API key
nor the configured base URL appears in it.

## Limitations

- FS-001 is one deliberately small coding domain.
- The healthy-control study contains four required synthetic trajectories, not
  a broad production corpus.
- Live sample size is three per arm and one model configuration.
- No live cycle occurred, so the semantic usefulness of an intervention to a
  real model remains unmeasured.
- Source mutations were sequential; concurrent multi-mutation causality was
  represented and would be classified ambiguous, but no such live batch
  occurred.
- Endpoint identity is retained as a hash; credentials and full URLs are
  intentionally excluded.

## Observation-only implementation follow-up

The first Shadow Mode implementation now exists without changing Runtime
behavior. Its internal Interface, invariants, and next-phase entry criteria are
documented in [Trajectory Shadow Analysis](trajectory-shadow-design.md).

The detector threshold was also tightened after the experiment review: CP3 is
only `CycleSuspected`; CP4 is required to prove the complete repeated
`A/B/A/B` Action path and emit `CycleConfirmed` at the next decision boundary.
