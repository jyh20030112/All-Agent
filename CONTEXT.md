# EJAgent Runtime

This context defines the language used to reason about an agent's execution,
its environment, and whether its trajectory is making verifiable progress.

## Objective and decision language

**Goal**:
The stable user-controlled outcome that the agent is expected to achieve.
_Avoid_: Plan, task step

**Requirement**:
One independently assessable part of a Goal whose satisfaction contributes to
completion.
_Avoid_: Plan step, action

**Constraint**:
A condition that must remain true while the Goal is pursued, including
conditions that can invalidate otherwise successful work.
_Avoid_: Preference, suggestion

**Plan**:
A revisable hypothesis about how to move from the current State toward the
Goal.
_Avoid_: Goal, truth

**Belief**:
An agent-held interpretation of the environment that may guide a Plan but is
not authoritative until supported by Evidence.
_Avoid_: Environment Fact, truth

**Action**:
One concrete operation proposed or performed by the agent, independent of the
Tool through which it is expressed.
_Avoid_: Tool, plan step

**Tool**:
A capability interface through which Actions can be performed or Observations
can be obtained.
_Avoid_: Action

**Actor**:
The decision-maker that proposes Actions and may issue a Completion Claim while
pursuing a Goal.
_Avoid_: Evaluator, controller

**Controller**:
The authority that admits, delays, rejects, or terminates proposed Actions
according to current State, policy, and resource limits.
_Avoid_: Actor, evaluator

## Environment language

**Observation**:
Raw output obtained from an environment interaction at a particular point in
time.
_Avoid_: Environment Fact, Progress

**Evidence**:
A traceable Observation or artifact used to support or refute a claim within a
declared scope.
_Avoid_: Belief, assertion

**Verifier**:
A reproducible procedure whose result provides scope-limited Evidence for a
Requirement or Constraint.
_Avoid_: Actor self-report, broad proof

**Environment Fact**:
An immutable, source-attributed assertion about the environment at a specific
checkpoint, with an explicit scope and Evidence reference.
_Avoid_: Tool result, model claim, timeless truth

**Fact Validity**:
The checkpoint-relative status that says whether an Environment Fact is
current, invalidated, stale, or not yet known to be current.
_Avoid_: Fact mutation, confidence

**Derived Fact**:
An Environment Fact produced by a reproducible deterministic transformation of
other Environment Facts.
_Avoid_: Assessment, model inference

**Assessment**:
An evaluator's goal-relative interpretation of Facts, such as Progress,
Regression, or a suspected cycle.
_Avoid_: Environment Fact

**Evaluator**:
The decision-maker that produces Assessments from Goal-relative Facts and
Evidence without inheriting the Actor's authority to declare them true.
_Avoid_: Actor, verifier

**State**:
A goal-relevant projection of Environment Facts that are valid at one
checkpoint and sufficient for a class of decisions.
_Avoid_: Conversation, complete world model

**State Delta**:
The Facts added, changed, invalidated, or confirmed between two States.
_Avoid_: Progress Delta

**State Fingerprint**:
A lossy comparable identity derived from a State for recurrence detection; it
is evidence of possible equivalence, not the State itself.
_Avoid_: State, Environment Fact

**Checkpoint**:
A named boundary at which the environment is observed and a State may be
compared or evaluated.
_Avoid_: Turn, arbitrary timestamp

**Decision Boundary**:
A point at which the Actor may choose another Action and therefore may need a
fresh Context projection.
_Avoid_: Every event, every checkpoint

**Context Projection**:
A disposable, decision-relevant view of the Goal, current State, Evidence, and
Assessments assembled for one Actor decision without changing Conversation.
_Avoid_: Event log, Conversation truth

## Trajectory language

**Trajectory**:
The ordered history of Actions, Observations, States, Assessments, and control
decisions produced while pursuing a Goal.
_Avoid_: Conversation history, tool log

**Task Progress**:
Verified improvement in the current State's satisfaction of Goal Requirements
while preserving Constraints.
_Avoid_: Activity, state change

**Epistemic Progress**:
New Evidence that materially narrows uncertainty or changes the justified next
decision without yet improving Requirement satisfaction.
_Avoid_: New thought, plan rewrite

**Progress Snapshot**:
An Assessment of current Requirement coverage, Evidence gain, Constraints,
Regressions, blockers, and cost at a Checkpoint relative to a Goal.
_Avoid_: Single completion percentage

**Progress Delta**:
The assessed change between two Progress Snapshots.
_Avoid_: State Delta

**Regression**:
The loss of previously verified Requirement satisfaction or the introduction
of a Constraint violation.
_Avoid_: Any failed action

**Non-progress Cycle**:
A recurring Action and State pattern that produces neither sustained Task
Progress nor justified Epistemic Progress while cost continues to increase.
_Avoid_: Tool repetition, retry

**Completion Claim**:
An actor's proposal that the Goal has been achieved.
_Avoid_: Completion

**Completion Audit**:
An evaluation of every Goal Requirement and Constraint against authoritative,
scope-appropriate Evidence.
_Avoid_: Plan completion, actor self-report
