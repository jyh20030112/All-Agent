# Agent Harness Architecture

## Status

EJAgent is an Agent Harness. This document defines the implemented pre-1.0
architecture and is normative for its contracts. The
[Harness overview](harness-overview.md) describes the project scope; the
[domain glossary](../CONTEXT.md) defines the shared terms. `RuntimeKernel` is
the existing name of the single-Run execution component within the Harness.

Implementation progress:

- Typed message, Run, Model, Tool, cancellation, usage, and JSON contracts are
  implemented in `ejagent.contracts`.
- `ejagent.kernel.RuntimeKernel` executes one Run over a private workspace and
  returns a deterministic Delta plus Audit records.
- `ejagent.harness.AgentHarness` owns single-agent resource lifecycle,
  cancellation, FIFO Run admission, snapshot recovery, and atomic
  compare-and-commit through the new `SessionStore` contract.
- Without a Store, `AgentHarness` keeps only process-local state for its own
  lifetime. Durable state has one built-in adapter.
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
- `JsonlSessionStore` provides locked, append-only durable commits with CAS,
  idempotent Run IDs, crash-tail recovery, and typed Conversation/Audit codecs.
- `OpenAIModelPort` translates typed Context and Tool values at the Provider
  seam. Function, composite, and MCP ToolExecutors share the Kernel Tool
  contract; `SkillsContextPipeline` contributes disposable skill instructions.
- `AnthropicModelPort` independently translates system instructions, content
  blocks, streamed tool input, and cache-aware usage from the Anthropic
  Messages protocol. Both Provider adapters pass the same Kernel-facing stream
  contract without adding Provider concepts to Core.
- An optional `TrajectoryMonitor` captures execution boundaries. The internal
  online implementation combines host evaluation with progress and cycle
  assessment; a composed Context pipeline can expose feedback to the next
  model decision. Completion advice is recorded without enforcing continuation.

## Scope

The Harness coordinates Context, capabilities, control, continuity, and
evaluation for an agent. The current `AgentHarness` instance owns one logical
agent and delegates single-Run execution to its Kernel:

```text
AgentHarness
  ├─ long-lived conversation state
  ├─ resource and control lifecycle
  ├─ context projection
  ├─ model and tool capabilities
  ├─ optional environment evaluation and trajectory feedback
  ├─ durable commit coordination
  └─ RuntimeKernel
       └─ one Run with deterministic message ordering
```

Multi-agent management and arbitrary mid-Run pause/resume are not implemented.
The supported controls are cancellation, steering at model-call safe points,
queued follow-ups, and continuation from a committed revision.

## Ownership Boundaries

### Kernel: single-Run execution

`RuntimeKernel` owns only Run-local state: turn counters, usage, repeat guards,
cancellation, the private message workspace, and temporary context. It receives
an immutable `RunSpec` and returns a `RunOutcome`. It never starts resources,
persists Sessions, or mutates Harness state.

### Harness: lifecycle, accepted state, and composition

The Harness owns conversation revisions, the last committed result, Provider
and Tool resource lifecycle, control queues, context policy, and SessionStore
commit coordination. Each Run receives a snapshot of the current revision;
per-Run limits and metadata are captured when that Run is admitted. The
configured Context and trajectory dependencies participate in execution
through their own interfaces.

### Host evaluator: domain truth

The host supplies environment access and Requirement/Constraint evaluation.
The Kernel identifies when a Tool batch has completed; the evaluator determines
what that batch changed in the application's domain. The trajectory analyzer
derives Assessments, and the Context adapter selects model-visible feedback.
This composition is a Harness capability even though each part has a separate
implementation and ownership boundary.

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

Only `RunStatus.COMPLETED` advances Conversation in the current
`SessionCommit` implementation. Failed, rejected, and cancelled Runs remain
auditable without promoting their Delta. There is no injectable commit policy
that changes this rule. A Conversation rollback does not undo Tool side effects.

## Extension Boundaries

Current extension points are:

1. `ModelPort` normalizes Provider requests and streams.
2. `ContextPipeline` builds ContextViews; `ContextCompactor` derives summaries.
3. `ToolExecutor` exposes Tool definitions and executes calls. The Kernel
   coordinates concurrent batches; domain authorization belongs to the adapter.
4. `TrajectoryMonitor` in `ejagent.kernel` observes semantic execution boundaries.
   The built-in evaluator and projection types remain internal to `_trajectory`.
5. `RunObserver` receives completed Run audits after the Store decision.
6. `SessionStore` commits revisions and durable Run records.
7. `ManagedResource` exposes lifecycle managed by the Harness.

`RunLimits` supplies turn, token, and repeated-call bounds. Budget and
termination behavior is implemented in the Kernel, and commit eligibility in
`SessionCommit`; no injectable `RunPolicy` or `ToolRuntime` interface exists.

Resources are started transactionally by the Harness and shut down in reverse
order. Kernel and Tool dispatch methods require ready dependencies.

## Tool Execution

All Tool calls from one model response execute concurrently. Results enter the
Conversation in the model's source order, while Audit completion events retain
actual timing. If one call fails at the Tool infrastructure seam or the Run is
cancelled, unfinished sibling calls are cancelled. Core has no serial mode and
does not retry Tools.

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

The Harness remains in one distribution. Module dependency direction
is enforced before considering separate Provider or backend packages.

## Feedback and Control Evolution

State continuity, Context composition, capabilities, and existing controls are
implemented parts of the Harness. Online trajectory feedback adds evidence to
the decision cycle while preserving these ownership boundaries.

The Kernel currently accepts a terminal text response even if the monitor
returns `completion_allowed=False`. ADR 0001 records the intended same-Run
feedback behavior for a future completion-enforcement policy; it is not enabled
by composing the monitor today. Automatic denial, forced replanning, and a
general domain verifier are likewise not current capabilities.

## Acceptance Criteria

- Kernel tests prove no mutation escapes before a Harness commit.
- Identical RunSpec and scripted dependencies produce identical committed
  message order.
- Durable commit failure cannot advance the active revision or report success.
- Observer failure cannot change a Run result.
- Context compaction is rebuildable from immutable Conversation history.
- Tool execution timing may vary, but Delta order remains deterministic.
- ModelPort and SessionStore implementations pass shared contract suites.
- Kernel modules do not import concrete Providers, Stores, MCP, or Skills.
- The Kernel observes trajectory through its protocol without importing the
  internal analyzer; model-visible feedback passes through Context.
- Domain evaluation, analysis, and control authority remain distinguishable.
