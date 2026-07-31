# Runtime Kernel and Agent Harness Design

## Status

This document defines the target architecture for the pre-1.0 EJAgent Core
refactor. It is normative for new contracts. Existing `BaseAgent`, dictionary
messages, and event-driven Session recording remain migration sources, not
constraints on the target design.

Implementation progress:

- Typed message, Run, Model, Tool, cancellation, usage, and JSON contracts are
  implemented in `ejagent.contracts`.
- `ejagent.kernel.RuntimeKernel` executes one Run over a private workspace and
  returns a deterministic Delta plus Audit records.
- `ejagent.harness.AgentHarness` owns single-agent resource lifecycle,
  cancellation, FIFO Run admission, snapshot recovery, and atomic
  compare-and-commit through the new `SessionStore` contract.
- `MemorySessionStore` provides an idempotent in-process adapter and rejects
  stale revisions and reused Run IDs with different content.
- Conversation recovery now uses an immutable `ConversationSnapshot`; durable
  Run facts are read separately as `RunAudit` values.
- Every model call receives a disposable `ContextView` from a
  `ContextPipeline`. Identity and derived-compaction implementations preserve
  committed history while permitting summaries and transient instructions.
- Steering is admitted by the Harness and consumed by the Kernel only at model
  call safe points; unapplied inputs remain auditable without entering
  Conversation.
- Follow-ups run as independent FIFO Runs. `RunObserver` delivery happens
  asynchronously after the Store decision, so observer latency and failure
  cannot alter execution or commit semantics.
- Durable legacy adapters and Session migration remain pending. The legacy
  execution path is not yet connected to the new Kernel.

## Scope

EJAgent Core contains two layers for one logical agent:

```text
AgentHarness
  ├─ long-lived conversation state
  ├─ resource and control lifecycle
  ├─ context projection
  ├─ durable commit coordination
  └─ RuntimeKernel
       └─ one deterministic model-tool Run
```

Multi-agent management and arbitrary mid-Run pause/resume are out of scope.
The supported controls are cancellation, steering at model-call safe points,
queued follow-ups, and continuation from a committed revision.

## Ownership Boundaries

### RuntimeKernel

The Kernel owns only Run-local state: turn counters, usage, repeat guards,
cancellation, the private message workspace, and temporary context. It receives
an immutable `RunSpec` and returns a `RunOutcome`. It never starts resources,
persists Sessions, or mutates Harness state.

### AgentHarness

The Harness owns conversation revisions, the last committed result, Provider
and Tool resource lifecycle, control queues, context policy, and SessionStore
commit coordination. Each Run receives a snapshot of the current revision.
Configuration changes take effect only on the next Run.

## Data Domains

Three data domains must remain distinct:

- **Conversation** contains immutable, typed messages that are valid input to a
  future Run.
- **Audit** is an append-only account of what actually happened, including
  partial failed or cancelled Runs and external side effects.
- **ContextView** is a disposable projection for one model request. It may use
  summaries, windows, Skills, and transient instructions without rewriting the
  Conversation or Audit.

Provider payload dictionaries are produced only by Provider adapters. Core
messages use a closed, Provider-neutral type union.

## Run Transaction

```text
committed revision N
       │ snapshot
       ▼
private RunWorkspace
       │ Kernel execution
       ▼
RunOutcome(result, delta, audit)
       │ compare-and-commit N
       ▼
committed revision N+1
```

`RunDelta.base_revision` prevents stale writes. A configured durable Store is a
correctness boundary: the Harness cannot report a committed completion until
the Store accepts the idempotent commit. Observer failures never substitute for
Store failures and do not alter execution.

Failed and cancelled Runs remain auditable but do not advance active
Conversation history by default. A Harness policy may explicitly promote an
eligible Delta.

## Extension Boundaries

New extension points are limited to narrow, ordered protocols:

1. `ModelPort` normalizes Provider requests and streams.
2. `ContextPipeline` contributes, transforms, and serializes ContextViews.
3. `ToolRuntime` validates, authorizes, schedules, and executes Tools.
4. `RunPolicy` decides budgets, retries, termination, and commit eligibility.
5. `RunObserver` observes events without changing Run results.
6. `SessionStore` commits revisions and durable Run records.

Resources are started transactionally by the Harness and shut down in reverse
order. Kernel and Tool dispatch methods require ready dependencies.

## Tool Semantics

Tool scheduling does not infer safety from implementation names. Definitions
declare effect, idempotency, and an optional concurrency key. Execution may be
concurrent only when semantics and policy permit it. Completion and
Conversation commit order always follow the model's source order; actual timing
is retained in Audit records. Non-idempotent Tools are never retried by Core.

## Failure Contract

Expected operational failures return structured outcomes. These include model
timeouts and rate limits, Tool failures, policy rejection, cancellation,
budgets, context overflow, and persistence failure. Exceptions are reserved for
invalid configuration, protocol violations, and broken invariants.

## Public API

`AgentHarness` is the primary entry point. `RuntimeKernel` has a narrow advanced
API for direct single-Run embedding. `BaseAgent` is removed rather than kept as
a compatibility execution path. Public contracts are intentionally small;
workspace mechanics, schedulers, commit coordinators, and state-machine phases
remain internal.

The initial refactor remains in one distribution. Module dependency direction
is enforced before considering separate Provider or backend packages.

## Migration Strategy

1. Add typed message and Run contracts with contract tests.
2. Implement the Kernel over a private workspace and deterministic Delta.
3. Implement the Harness, resource lifecycle, controls, and atomic commit.
4. Split Conversation, Audit, and ContextView; make compaction derived.
5. adapt Memory/JSONL Stores and provide legacy Session decoding/migration.
6. Adapt OpenAI, MCP, Skills, examples, and documentation.
7. Replace the public API and delete the old execution path.
8. Validate `ModelPort` with a second, materially different Provider protocol.

Intermediate steps may coexist on a development branch, but no release should
expose two competing execution or commit semantics.

## Acceptance Criteria

- Kernel tests prove no mutation escapes before a Harness commit.
- Identical RunSpec and scripted dependencies produce identical committed
  message order.
- Durable commit failure cannot advance the active revision or report success.
- Observer failure cannot change a Run result.
- Context compaction is rebuildable from immutable Conversation history.
- Tool execution timing may vary, but Delta order remains deterministic.
- ModelPort and SessionStore implementations pass shared contract suites.
- Legacy Session fixtures either migrate losslessly or fail with a typed,
  actionable migration error.
- Kernel modules do not import concrete Providers, Stores, MCP, or Skills.
