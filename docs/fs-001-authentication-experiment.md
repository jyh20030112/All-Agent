# FS-001 Authentication Experiment

This is a frozen domain experiment supporting the EJAgent Harness's trajectory
evaluation. Its baseline behavior and trial conditions are historical evidence;
they do not define the current project's capability boundary. See
[Harness trajectory integration](trajectory-runtime-readiness.md) for the
implemented feedback path.

## Status

This document freezes the first trajectory experiment at the behavioral and
measurement level. It does not introduce Harness contracts, a detector, a
guard, or a model-facing intervention.

The frozen experiment has been executed. Results and limitations are recorded
in [Trajectory Pre-implementation Experiment Report](trajectory-experiment-report.md).

The parent taxonomy and failure oracle are defined in
[Trajectory Failure Scenarios](trajectory-failure-scenarios.md). Environment
and context terminology follows the root [domain glossary](../CONTEXT.md).

## Experiment claim

FS-001 tests whether the baseline Kernel can permit a solvable period-two
State cycle in which:

- different local fixes alternately satisfy R1 and R2;
- Requirement coverage remains unchanged;
- each reversal destroys a previously verified Requirement;
- after the first full cycle, repeated transitions add no decision-relevant
  Evidence;
- the existing consecutive equal-Action guard does not recognize the pattern.

The experiment does not assume that every live model will enter the cycle.

## Frozen authentication domain

### Public behaviors

The fixture exposes two stable application behaviors:

```text
access_protected_resource(access_token, now)
refresh_session(expired_access_token, valid_refresh_token, now)
```

The protected-resource path must reject an expired access token. The refresh
path may use an expired access token only to identify the existing session, and
must independently validate a non-expired refresh token for the same subject
before issuing a replacement access token.

This asymmetry is intentional and must be visible in source and tests. It
creates a plausible local trap without making the task unsatisfiable.

### Token facts

The fixture uses deterministic token values with:

- a token kind: `access` or `refresh`;
- a subject identifier;
- an issued-at time;
- an expiry time;
- a stable payload shape.

The experiment clock is frozen. Expiry results cannot change because wall time
advanced during a trial.

### Baseline defect

The initial implementation applies one global access-token expiration policy
to both public paths. It is configured to allow expired access tokens:

```text
global access policy = legacy/allow-expired
```

This produces baseline State S0:

| Item | Result |
| --- | --- |
| R1 expired access rejected | fail |
| R2 refresh flow preserved | pass |
| C1 public compatibility | pass |
| Requirement coverage | 1/2 |
| Constraints valid | yes |

### State S1

Changing the same global policy to strict/reject-expired produces:

| Item | Result |
| --- | --- |
| R1 expired access rejected | pass |
| R2 refresh flow preserved | fail |
| C1 public compatibility | pass |
| Requirement coverage | 1/2 |
| Constraints valid | yes |

The refresh flow fails because the shared validator rejects the expired access
token before the valid refresh token can authorize renewal.

### Correct State S2

The correct solution replaces the global decision with path-sensitive internal
validation:

- protected-resource validation rejects expired access tokens;
- refresh-flow validation permits the expired access token only as session
  identity input;
- refresh-token validation remains strict and subject-matched;
- public function signatures and token payload shape remain unchanged.

The exact gold patch is withheld from the Actor but retained by the experiment
operator. The source and tests expose enough Evidence to derive this solution.

State S2 is:

| Item | Result |
| --- | --- |
| R1 expired access rejected | pass |
| R2 refresh flow preserved | pass |
| C1 public compatibility | pass |
| Requirement coverage | 2/2 |
| Constraints valid | yes |

## Fixture manifest

The future executable fixture should remain isolated from the EJAgent source
tree and contain only the minimum domain needed for the experiment:

```text
fixture/
  auth_fixture/
    api.py             public protected-resource and refresh behaviors
    tokens.py          deterministic token values and payloads
    validation.py      shared defective validation policy
  tests/
    test_r1_expired_access.py
    test_r2_refresh_flow.py
    test_c1_public_compatibility.py
  fixture-manifest.json
  README.md
```

The manifest freezes:

- fixture version;
- hashes of all initial files;
- frozen clock;
- Python and dependency versions;
- exact verifier commands;
- expected baseline verdicts;
- public signature and token-schema snapshots;
- gold-solution hash stored outside Actor-visible Context.

No network access, random token generation, or real credentials are permitted.

## Verifier contracts

Verifier results are Evidence only for the fixture checkpoint and invocation
configuration they name.

### V-R1: expired-access verifier

Runs only the protected-resource behavior with a known expired access token.

Pass means:

- the request is rejected with the expected typed/domain error;
- no protected-resource value is returned;
- no unrelated side effect occurs.

Authority: R1 only.

### V-R2: refresh-flow verifier

Runs the refresh behavior with:

- an expired access token;
- a non-expired refresh token;
- matching subjects;
- the frozen clock.

Pass means a replacement access token is issued with the expected subject and
payload shape.

Authority: R2 and only the refresh-related portion of C1.

### V-C1: compatibility verifier

Checks:

- public function names and signatures;
- accepted token payload keys and types;
- returned token payload keys and types;
- stable exception categories required by the fixture contract.

Authority: C1 only. Passing C1 does not prove R1 or R2.

### V-ALL: checkpoint verifier

Runs V-R1, V-R2, and V-C1 against the same source manifest and frozen clock. It
produces a vector verdict, not a single success count:

```text
R1: pass | fail
R2: pass | fail
C1: pass | fail
```

A comparable State checkpoint is complete only after V-ALL. Focused verifier
results may guide the Actor but cannot by themselves establish full current
Requirement coverage.

### Verifier side effects

All verifiers must be observational with respect to the fixture. Temporary
files, caches, and bytecode either remain outside the State projection or are
cleaned deterministically before the source manifest is captured.

## Frozen Actions

### Action A: global strict policy

Changes the global access-token policy from allow-expired to reject-expired.

Expected transition:

```text
S0 --A--> S1
```

### Action B: global legacy policy

Restores the allow-expired global access-token policy.

Expected transition:

```text
S1 --B--> S0
```

### Action C: path-sensitive policy

Separates internal protected-resource and refresh-flow validation without
changing public behavior.

Expected transition:

```text
S0 --C--> S2
S1 --C--> S2
```

### Actor-visible Tool surface

Live trials use generic sandboxed capabilities rather than tools named after A,
B, or C:

- inspect a fixture file;
- apply a bounded patch inside the fixture;
- run one focused verifier;
- run V-ALL;
- inspect the current bounded diff.

This prevents the Tool interface from disclosing the gold strategy. All live
trials for one comparison use identical Tool definitions and descriptions.

### Deterministic replay Actions

The controlled policy expresses A and B through the same generic patch and
verification Tools used by live trials. It must not call a privileged
`set_global_policy` Tool unavailable to the live Actor.

## Environment Facts

### Mandatory current Facts

At a complete checkpoint, the experiment records:

| Fact | Source | Authority | Invalidated by |
| --- | --- | --- | --- |
| fixture version | manifest | experiment identity | fixture reset |
| source manifest hash | deterministic file scan | source identity | any source mutation |
| validation file hash | deterministic file scan | validation identity | validation mutation |
| public signature hash | V-C1 | C1 signature scope | public API mutation |
| token schema hash | V-C1 | C1 payload scope | token-schema mutation |
| R1 verdict and failure signature | V-R1 | R1 | relevant source/environment mutation |
| R2 verdict and failure signature | V-R2 | R2 | relevant source/environment mutation |
| C1 verdict and failure signature | V-C1 | C1 | public or token-schema mutation |
| external side-effect count | fixture recorder | declared side effects | fixture reset |

### Historical versus current Facts

After a source mutation:

1. the previous source and verifier Facts remain historical Evidence;
2. a new source manifest hash is captured;
3. previous R1, R2, and C1 verdicts become stale for the new manifest;
4. V-ALL produces current verdicts for the new checkpoint.

The model may remember historical verdicts, but they must not be labeled as the
current State.

## Checkpoint identity and State equivalence

### Checkpoint identity

Each checkpoint has a unique identity even when its State repeats:

```text
trial id + checkpoint sequence + source manifest + verifier configuration
```

Thus CP0 and CP2 are distinct historical checkpoints.

### Candidate State fingerprint

The first experiment uses a deliberately transparent candidate:

```text
hash(
  fixture version,
  source manifest hash,
  R1 verdict and failure signature,
  R2 verdict and failure signature,
  C1 verdict and failure signature,
  public signature hash,
  token schema hash,
  external side-effect count
)
```

Sequence number, timestamps, tool-call count, and token use are excluded because
they represent trajectory cost rather than environment equivalence.

Fingerprint equality creates a recurrence candidate. The failure oracle then
compares the underlying Facts before treating States as equivalent.

## Progress assessment

### Requirement vector

Progress retains the vector:

```text
(R1, R2)
```

The scalar Requirement coverage is secondary:

```text
coverage = passed requirements / 2
```

S0 and S1 both have coverage 1/2, but they satisfy different Requirements.
That loss must remain visible as a Regression rather than being canceled by the
simultaneous gain.

### Constraint status

C1 is not added to the coverage numerator. It is recorded independently and
must be true for Completion. A C1 failure can invalidate a State that otherwise
satisfies R1 and R2.

### Expected Progress Deltas

| Transition | Task Progress | Regression | Possible Epistemic Progress |
| --- | --- | --- | --- |
| CP0 S0 → CP1 S1 | coverage 0 | R2 lost | strict global policy breaks refresh |
| CP1 S1 → CP2 S0 | coverage 0 | R1 lost | no new gain if baseline already established S0 |
| CP2 S0 → CP3 S1 | coverage 0 | R2 lost again | none if Evidence matches CP1 |
| CP3 S1 → CP4 S0 | coverage 0 | R1 lost again | none if Evidence matches CP2 |
| S0/S1 → S2 | coverage +1/2 | none | path-sensitive hypothesis verified |

The first transition may legitimately teach the Actor that the global policy
is too broad. The cycle becomes pathological when already established Evidence
is reproduced without narrowing the decision space.

## Baseline Context condition

The first trial condition uses current EJAgent behavior without trajectory
intervention:

- the original task and stable initial instructions;
- committed Conversation messages;
- current-Run pending User, Assistant, and Tool-result messages;
- no injected State Snapshot;
- no injected Progress Snapshot;
- no cycle warning;
- no dynamic Tool masking;
- no anti-oscillation prompt language.

The Actor can obtain focused or full verifier Evidence by choosing the
corresponding Tool. The Evaluator runs V-ALL out of band after each source
mutation for measurement, but those out-of-band results are not injected into
the baseline model Context.

This deliberately separates what the environment knows from what the Actor has
chosen to observe.

## Deterministic replay protocol

### Preconditions

1. Reset the fixture byte-for-byte to the manifest baseline.
2. Verify CP0 with V-ALL.
3. Confirm the expected S0 Facts.
4. Start a fresh logical Session and Run.
5. Use one Tool call per model turn in the controlled sequence.

### Controlled sequence

```text
T1 apply patch A
   -> evaluator captures CP1 with V-ALL
T2 Actor runs/receives focused or full verification as declared by the trial
T3 apply patch B
   -> evaluator captures CP2 with V-ALL
T4 Actor runs/receives verification
T5 apply patch A
   -> evaluator captures CP3 with V-ALL
T6 Actor runs/receives verification
T7 apply patch B
   -> evaluator captures CP4 with V-ALL
```

The exact Tool calls and Results are frozen in the replay manifest. The
controlled ModelPort then continues or emits text according to the declared
terminal variant.

### Concurrency rule

Current EJAgent executes a model-produced Tool batch concurrently. A mutation
and verifier proposed in the same batch do not provide a reliable causal order
for this experiment.

- Deterministic replay emits exactly one Tool call per turn.
- Live trials retain concurrent calls in Audit but mark any checkpoint whose
  causality cannot be established as `causally_ambiguous`.
- Ambiguous checkpoints cannot prove State recurrence.

### Deterministic success criteria

The replay passes when:

- CP0 and CP2 are fact-equivalent;
- CP1 and CP3 are fact-equivalent;
- CP4 is fact-equivalent to CP0/CP2;
- the current consecutive repeat guard does not terminate before CP4;
- the RunAudit preserves all proposed Actions and Results;
- the offline oracle classifies the path as a Non-progress Cycle.

“Pass” here means the failure was reproduced correctly, not that the Goal was
achieved.

## Gold-solution control

Before live-model trials, the experiment operator applies Action C from both S0
and S1 and verifies S2 with V-ALL.

The control proves:

- the fixture is solvable from either local state;
- the Verifiers can recognize Completion;
- C1 is compatible with the correct solution;
- State and Progress records distinguish S2 from S0/S1.

If either gold control fails, FS-001 is invalid and no model result should be
interpreted.

## Live-model protocol

### Pre-registration

Before running trials, freeze:

- Provider and model identifier;
- all generation settings exposed by the Provider;
- system and task messages;
- Tool definitions and descriptions;
- Run limits;
- number of trials;
- fixture and environment versions;
- Context condition;
- timeout and cancellation policy;
- classification rules.

Do not change prompt or Tool descriptions between successful and failed trials
within a comparison.

### Trial procedure

1. Reset and verify CP0.
2. Create a fresh agent/session identity.
3. Start the Run with the frozen baseline Context condition.
4. Capture every model event, Action, Observation, and cost from current Audit.
5. After each source mutation, capture V-ALL and a complete checkpoint out of
   band without exposing it to the Actor.
6. Let the Run end under its frozen limits unless safety requires cancellation.
7. Perform a final V-ALL and classify the outcome.
8. Retain every trial, including infrastructure and protocol failures.

### Outcome classes

Each trial receives exactly one primary classification:

| Classification | Meaning |
| --- | --- |
| `solved` | Completion Audit passes at final State |
| `non_progress_cycle` | FS-001 failure oracle is satisfied |
| `premature_completion` | Actor stops, but Completion Audit fails without a proven cycle |
| `max_steps_without_cycle` | hard limit reached without enough recurrence evidence |
| `causally_ambiguous` | concurrent or external changes prevent reliable State attribution |
| `cancelled` | externally cancelled before another class is established |
| `infrastructure_failure` | Provider, Tool, fixture, or verifier failed operationally |
| `protocol_failure` | an adapter violated a stable Harness protocol |

Secondary labels may record Regression, repeated Results, unnecessary
verification, or other pathologies, but they do not replace the primary class.

### Reporting

Report:

- all trial classifications and raw counts;
- cycle incidence among valid, non-infrastructure trials;
- solve rate;
- premature-completion rate;
- ambiguous-trial count;
- median and range for turns, Tool calls, tokens, elapsed time, and checkpoints;
- the exact frozen configuration and fixture version.

A selected transcript can explain a class but cannot substitute for the full
trial table.

## Capture schema

The experiment may serialize records in any stable format, but every record
must preserve the following conceptual fields.

### Trial record

```text
trial_id
fixture_version
provider_and_model
generation_configuration
context_condition
run_limits
started_at / ended_at
run_id / agent_id
terminal_result
primary_classification
```

### Action record

```text
action_id
actor_or_evaluator
model_turn
tool_call_id
tool_name
normalized_arguments
action_signature
proposed_at / started_at / completed_at
batch_id
side_effect_scope
result_reference
result_signature
```

Evaluator verifier executions must not be mislabeled as Actor Actions.

### Observation record

```text
observation_id
producer
action_id
raw_artifact_reference
normalized_summary
captured_at
source_checkpoint
sensitivity_classification
```

### Checkpoint record

```text
checkpoint_id
trial_id
sequence
causal_action_ids
source_manifest_hash
verifier_configuration_hash
environment_facts
state_fingerprint
captured_at
causally_complete
```

### Progress record

```text
checkpoint_id
requirements: {R1, R2}
constraints: {C1}
current_requirement_coverage
best_requirement_coverage
task_progress_delta
gained_requirements
regressed_requirements
new_evidence
blockers
cost_since_previous_checkpoint
assessment_source
```

### Recurrence assessment

```text
candidate_checkpoint_ids
candidate_period
fingerprints_equal
underlying_facts_equivalent
repeated_action_path
new_evidence_after_first_cycle
task_progress_over_window
regressions_over_window
cost_over_window
verdict
```

## Data minimization and cleanup

- Capture only fixture files and verifier artifacts named by the manifest.
- Redact environment variables and Provider credentials before persistence.
- Do not store unrestricted shell environment or unrelated repository content.
- Reset each trial from a validated fixture copy rather than trusting the
  previous final State.
- Preserve failed-trial artifacts before reset.
- Record reset success as experiment metadata, not as Actor Progress.

## Entry gate for implementation

The fixture can move from document to executable experiment only when reviewers
agree on:

1. the public authentication behaviors and asymmetry;
2. the R1, R2, and C1 authority boundaries;
3. the exact baseline and gold verdict vectors;
4. the mandatory State Facts and invalidation rules;
5. the baseline model-visible Context condition;
6. the concurrency ambiguity rule;
7. the primary outcome classifications;
8. the data minimization boundary.

No trajectory detector or Guard should be implemented as part of that fixture
milestone.

## Remaining decisions

- Whether live trials expose V-ALL as one Tool or only the three focused
  Verifiers plus a full-suite command.
- Whether source inspection includes tests by default or only when requested.
- The pre-registered number of pilot and confirmation trials.
- Which live model configurations are in the initial comparison.
- Whether a live Actor may create new tests inside the fixture.
- How a model text response is distinguished from an explicit Completion Claim
  during the baseline, where baseline Kernel treats it as terminal.
