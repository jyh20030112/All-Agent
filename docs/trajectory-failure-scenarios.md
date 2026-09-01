# Trajectory Failure Scenarios

## Status

This is a pre-implementation experiment specification. Its purpose is to make
trajectory failures reproducible before EJAgent introduces new runtime policy,
state, progress, or context contracts.

The first experiment deliberately targets a failure that the current
consecutive repeated-call guard cannot identify. No detector behavior is
prescribed here.

FS-001, its healthy controls, and live comparison arms have now been executed.
See [Trajectory Pre-implementation Experiment Report](trajectory-experiment-report.md).

The frozen execution contract for FS-001 is defined in
[FS-001 Authentication Experiment](fs-001-authentication-experiment.md).

## Questions

The experiments answer four questions in order:

1. Can the current Runtime enter a repeatable, solvable non-progress cycle?
2. Which observations are necessary to distinguish the failure from healthy
   repeated work?
3. Which environment facts remain authoritative at each checkpoint?
4. What is the smallest model-visible projection that could support a better
   next decision?

## FS-001: Alternating Authentication Fix

### Goal

Correctly reject expired access tokens while preserving the refresh-token
flow.

### Requirements and constraints

- **R1 — expired access token**: an expired access token is rejected by the
  protected endpoint.
- **R2 — refresh flow**: a valid refresh token can still obtain a replacement
  access token.
- **C1 — public compatibility**: the public authentication API and token
  payload shape remain compatible.
- **Completion**: independent verifiers for R1, R2, and C1 all pass against the
  same environment checkpoint.

### Environment shape

The fixture must contain:

- a shared validation path with a tempting global strict/legacy behavior;
- separate access-token and refresh-token call paths;
- focused verifiers for R1 and R2;
- a compatibility verifier for C1;
- a correct path-sensitive implementation that is discoverable from the
  source and tests.

The task must be solvable. An unsatisfiable fixture would test failure handling,
not trajectory control.

### Baseline State

```text
State S0
  R1 expired-access verifier: failed
  R2 refresh-flow verifier: passed
  C1 compatibility verifier: passed
  requirement coverage: 1/2
  constraints valid: yes
```

### Tempting local actions

**Action A — enable strict validation globally**

```text
S0 --A--> S1

State S1
  R1 expired-access verifier: passed
  R2 refresh-flow verifier: failed
  C1 compatibility verifier: passed
  requirement coverage: 1/2
  constraints valid: yes
```

**Action B — restore legacy validation globally**

```text
S1 --B--> S0
```

The correct Action C changes validation by call path instead of switching the
global behavior. Its exact patch is part of the hidden fixture oracle, but the
source and verifier evidence required to discover it remain available to the
agent.

### Expected pathological trajectory

Checkpoints are captured after a mutation and its verification:

| Checkpoint | Action since previous checkpoint | R1 | R2 | C1 | Coverage | State |
| --- | --- | --- | --- | --- | --- | --- |
| CP0 | baseline | fail | pass | pass | 1/2 | S0 |
| CP1 | A: strict globally | pass | fail | pass | 1/2 | S1 |
| CP2 | B: legacy globally | fail | pass | pass | 1/2 | S0 |
| CP3 | A: strict globally | pass | fail | pass | 1/2 | S1 |
| CP4 | B: legacy globally | fail | pass | pass | 1/2 | S0 |

The meaningful cycle is `S0 → S1 → S0`, even if the concrete Tool sequence
also contains reads and verifier commands.

### Failure oracle

FS-001 is a reproduced trajectory failure when all of the following hold:

1. R1, R2, and C1 are never simultaneously verified at one checkpoint.
2. At least one State equivalent to S0 and one State equivalent to S1 recur.
3. The alternating path repeats after its first complete cycle.
4. Requirement coverage does not exceed the baseline best coverage, while the
   satisfied Requirement alternates between R1 and R2.
5. No new Evidence after the first cycle justifies repeating A or B.
6. Tool calls, tokens, turns, or elapsed time continue to increase.
7. The Run continues until an unrelated hard limit, cancellation, or external
   intervention rather than recognizing the cycle itself.

The oracle must compare the underlying facts, not only a fingerprint.

### Why current protection is insufficient

The current repeat guard compares each Tool name and normalized arguments only
with the immediately preceding signature. Alternating A and B resets its
counter, so a period-two cycle is not rejected. It also evaluates the Action
before its Result or resulting environment State exists.

This is an expected baseline observation, not yet a proposal to replace the
guard.

## Reproduction modes

### Deterministic policy replay

A controlled ModelPort emits the intended alternating decisions. This mode
must be deterministic and proves that the Runtime and its existing controls
permit the pathological trajectory.

It does not prove that a production model is likely to choose the trajectory.

### Live-model trial

One or more real Provider configurations run the same fixture from the same
baseline. Every trial records its model configuration, sampling configuration,
initial State, final State, Run outcome, and complete Evidence trail.

The report must include all trials and the observed incidence rate. A single
selected failure may establish existence but must not be presented as a
frequency estimate.

### Required artifacts

Each trial must retain:

- immutable Run and configuration identifiers;
- current EJAgent RunAudit records;
- normalized Action and Observation records;
- raw verifier results and exit status;
- environment checkpoint identifiers;
- relevant State Facts at each checkpoint;
- current and best Requirement coverage;
- token, tool-call, turn, elapsed-time, and terminal-reason data;
- any external cancellation or human intervention.

Secrets and unrelated environment content must not be captured.

## Healthy controls

A useful detector must preserve these trajectories. They are part of the
experiment, not optional follow-up work.

### HC-001: Productive polling

```text
poll(job) -> 10%
poll(job) -> 60%
poll(job) -> complete
```

The Action signature repeats, but each Observation and State demonstrates
external progress.

### HC-002: Productive edit-verify loop

```text
edit -> verify: 8 failures
edit -> verify: 5 failures
edit -> verify: 2 failures
edit -> verify: 0 failures
```

The Tool and semantic Action classes repeat while Requirement satisfaction
increases.

### HC-003: Evidence-gaining exploration

```text
inspect A -> hypothesis A excluded
inspect B -> target narrowed to B
inspect C -> change boundary confirmed
```

The external State may remain unchanged, but each Observation supplies
decision-relevant Evidence and therefore Epistemic Progress.

### HC-004: Legitimate retry

A retry follows a transient failure and succeeds after an external condition
changes. The repeated Action is justified by new Evidence or a verified wait
condition.

## Comparison record

Every failing and healthy trial should be reducible to the same conceptual
record:

```text
Checkpoint
  Goal and active Requirements
  proposed and executed Action
  raw Observation references
  valid Environment Facts
  State and State Delta
  Progress Snapshot and Progress Delta
  cost since previous Checkpoint
```

This shared record prevents the failure case from defining a detector that
cannot represent healthy repetition.

## Experiment order

1. Freeze the fixture, requirements, verifiers, and baseline State.
2. Capture one successful Action C trajectory to prove solvability.
3. Capture deterministic A/B replay to prove runtime permissiveness.
4. Capture HC-001 through HC-004.
5. Run live-model trials without changing prompts between outcomes.
6. Compare which Facts separate the failing and healthy trajectories.
7. Only then propose a Progress evaluator or trajectory policy.

## Open questions

- Which State Facts are necessary and sufficient to distinguish S0 and S1?
- Should source content, verifier results, or both define checkpoint identity?
- When does a repeated inspection count as new Evidence rather than no
  progress?
- How many completed cycles constitute operational failure?
- Which costs belong to the cycle window when tools execute concurrently?
- How should external changes between checkpoints invalidate previous Facts?
