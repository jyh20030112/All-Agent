# EJAgent Core

[English](README.md) | [简体中文](README_zh-CN.md)

EJAgent Core is an OpenAI-first Agent Harness Core built around eight capability
areas: execution, control, extensibility, safety, context management, durable
history, branching, and observability. It provides the runtime mechanisms for
state, orchestration, context construction, tool dispatch, middleware, MCP, and
skills, while derived agents own concrete tools such as shell, file editing,
Git, or explicit completion.

Requires Python 3.12 or newer.

## Core capabilities

- Stateful `BaseAgent` with persistent conversation history and `reset()`
- Provider-neutral `ModelAdapter` boundary with an OpenAI-compatible adapter
- Public `AgentOrchestrator` for the provider-tool loop
- Structured `AgentRunResult`, `RunStatus`, and `StopReason`
- Explicit `RuntimePolicy` for loop and completion behavior
- `AgentContextBuilder` for non-mutating per-turn context projection
- OpenAI-first canonical `ToolDefinition` with legacy dictionary compatibility
- Composable `BaseHandler` and `MethodToolHandler` tool contracts
- `ToolRuntime` lifecycle, routing, middleware, and repeat-call protection
- Generic `ToolMiddleware` interception
- Optional fail-closed Tool Execution Policy middleware with approval and
  atomic per-run call limits
- Optional local JSON Schema argument validation before policy and handlers
- Structured, cancellable Tool Progress events
- Provider-neutral token Usage and per-run budget guards
- Context pressure estimates, independent window budgets, and non-mutating
  compaction preparation
- Explicit, cancellable compaction through a pluggable `Compactor`, canonical
  `SummaryEntry`, and resumable Session snapshots
- Opt-in automatic compaction on context pressure and one safe recovery attempt
  for provider-normalized context overflow
- Versioned Session serialization, append-only JSONL journals, and explicit
  cross-process restoration
- Optional MCP integration through `McpToolHandler`
- Local skill discovery, metadata projection, and explicit context activation

The core intentionally does not provide Bash, Git, filesystem, approval UI, or
finish tools. Those belong to a derived agent such as a CodeAgent.

This core-boundary change removes the former `BashHandler`, `GitDiffHandler`,
`FinishHandler`, `HumanApproval`, and `BashApprovalMiddleware` public exports.
Derived agents should provide equivalent implementations when needed.

## Installation

```bash
pip install ejagent-core
# or, for a source checkout: uv sync
```

MCP support is optional. Install its extra only when the agent uses MCP:

```bash
uv sync --extra mcp
# or: pip install "ejagent-core[mcp]"
```

Version 0.6.0 renames the PyPI distribution to `ejagent-core` and the Python
import package to `ejagent`:

```python
from ejagent import BaseAgent
```

The former `simagentplg` import is not retained as a compatibility alias.
Existing JSONL Sessions remain readable; legacy internal message metadata is
filtered from provider requests during restoration.

## Configuration

Copy `.env.example` to `.env` and provide model credentials:

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-flash
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.7
LLM_INCLUDE_USAGE=true
```

`ModelConfig` belongs to `OpenAIModelAdapter`, rather than to `BaseAgent`.
Configuration can also be supplied directly:

```python
from ejagent import ModelConfig

config = ModelConfig(
    model="deepseek-v4-flash",
    api_key="sk-xxxxxxxx",
    base_url="https://api.deepseek.com",
)
```

Other model providers can integrate with the core by implementing
`ModelAdapter.complete()` and optionally overriding `ModelAdapter.stream()`.
The adapter owns provider client creation, response normalization, streaming,
and optional startup/shutdown resources; `BaseAgent` only consumes
provider-neutral stream events and the normalized `AssistantMessage` contract.

## Plain agent

Conversation history is preserved across calls:

```python
from ejagent import BaseAgent, ModelConfig, OpenAIModelAdapter

agent = BaseAgent(
    OpenAIModelAdapter(ModelConfig.from_env()),
    agent_id="tutor",
    system_prompt="You are a concise Python tutor.",
)

first = await agent.runtime(task="Remember that I prefer Python.")
second = await agent.runtime(task="Which language do I prefer?")

agent.reset()
await agent.shutdown()
```

Calls on the same agent are serialized to protect conversation state.

## Structured runs

`run()` exposes the core result protocol:

```python
result = await agent.run(task="Explain the repository architecture.")

print(result.status)
print(result.stop_reason)
print(result.turns)
print(result.output)
```

`runtime()` remains a compatibility wrapper. It returns `result.output` for a
completed run and raises `AgentRunError` for failed, rejected, or cancelled
runs.

### Cancelling a run

Each run owns an independent cancellation token. `abort()` requests
cancellation without waiting, while `wait_for_idle()` settles only after the
terminal event and all awaited event sinks have completed:

```python
import asyncio

run = asyncio.create_task(agent.run(task="Perform a long operation."))

agent.abort("stopped by user")
await agent.wait_for_idle()
result = await run
```

An externally aborted run returns `RunStatus.CANCELLED` with
`StopReason.EXTERNAL_ABORT`. The same agent can be reused for another run.
Model adapters, tool middleware, and tool handlers receive the run's
`CancellationToken`; long-running handlers should also use `try/finally` to
release resources such as subprocesses.

### Steering an active run

`steer()` queues guidance for the active Run without interrupting an in-flight
model response or Tool call:

```python
receipt = await agent.steer(
    "Do not modify the database; produce a migration plan instead."
)
if not receipt.accepted:
    print(receipt.status)
```

Each Run owns a bounded FIFO queue; configure it with
`BaseAgent(..., steering_queue_capacity=16)`. Submission returns a typed
`ControlReceipt` with `ACCEPTED`, `AGENT_IDLE`, `QUEUE_FULL`, or `RUN_CLOSING`
instead of silently dropping input. Accepted guidance is consumed immediately
before the next Provider context is finalized, including a model retry after
Context Overflow recovery. A Tool that has already started always settles
before Steering reaches the model.

Applied guidance becomes a normal User message in Agent history and emits
`SteeringApplied`; `SessionRecorder` stores it as `steering_applied`. If the Run
ends or is cancelled before another model-call safe point, the pending input is
not persisted and emits `SteeringDiscarded`. An accepted receipt therefore
means “queued”, while these events expose the final applied/discarded outcome.

### Follow-up runs

`follow_up()` queues an independent Run behind the active Run chain and returns
a waitable handle:

```python
import asyncio

initial = asyncio.create_task(agent.run(task="Inspect the deployment failure."))
await asyncio.sleep(0)  # allow the Run chain to become active
follow_up = await agent.follow_up("Now propose the smallest safe fix.")

initial_result = await initial
if follow_up.accepted:
    follow_up_result = await follow_up.wait()
```

Each Follow-up receives its own `run_id`, lifecycle events, Session Run, and
`AgentRunResult`. It starts only after the previous `AgentFinished` and every
awaited Event Sink have settled. A Run-chain gate keeps already-waiting direct
`run()`, `compact()`, and lifecycle operations from interleaving ahead of the
Follow-up FIFO.

Configure the queue with `follow_up_queue_capacity`. Rejected handles expose an
immediate `ControlReceipt` with `AGENT_IDLE`, `QUEUE_FULL`, or `RUN_CLOSING`, and
`wait()` raises `FollowUpRejectedError`. By default, a failed or cancelled Run
discards remaining accepted handles with `FollowUpDiscardedError`; opt into
`FollowUpFailurePolicy.CONTINUE` to keep processing. Cancelling one caller's
`wait()` does not cancel the queued Run. `shutdown()` discards pending
Follow-ups, while `wait_for_idle()` includes accepted Follow-ups and their
terminal sinks.

Pending Follow-ups remain process-local. `SessionRecorder` writes `run_started`
only when a Follow-up actually begins, so the queue is not presented as a
durable job system.

### Continue existing history

`continue_run()` starts a distinct Run from committed history without appending
another User message:

```python
if agent.can_continue:
    result = await agent.continue_run()
else:
    print(agent.continue_rejection_reason)
```

Continue allocates a new `run_id`, emits `AgentContinued`, and is persisted as
`run_continued`. Session Runs expose `SessionRunIntent.TASK` or
`SessionRunIntent.CONTINUE`, with `task=None` for Continue. Restoring a finished
Session also restores the latest terminal result, so a new Agent instance can
continue the checked-out history.

Only explicit safe terminal reasons are continuable. Active Agents, Shutdown,
missing previous Runs, unsupported failure/cancellation reasons, and unmatched
Tool Calls raise `ContinueRejectedError` with a typed
`ContinueRejectedReason`. Continue reuses Cancellation, Steering safe points,
Follow-up chaining, automatic Compaction, terminal Event Sink barriers, and
`wait_for_idle()` semantics.

### Behavior Hooks

`BehaviorHook` controls whether a non-terminal full Turn may advance to the
next Provider request. It is deliberately separate from read-only Event Sinks
and Tool Middleware:

```python
from ejagent import BehaviorDecision


class StopAfterFirstToolTurn:
    async def after_turn(self, snapshot, *, cancellation):
        if snapshot.turn >= 1:
            return BehaviorDecision.stop("paused at a safe Turn boundary")
        return None


agent = BaseAgent(
    model,
    agent_id="controlled-agent",
    behavior_hooks=[StopAfterFirstToolTurn()],
)
```

The decision point runs after `TurnCompleted` work, including every committed
Tool Result, and before another `TurnStarted` or Provider request. Hooks receive
a detached `TurnSnapshot` plus the Run's `CancellationToken`; they never receive
mutable `AgentState`. Multiple Hooks are awaited in declaration order, and the
first STOP short-circuits the rest.

STOP completes the Run with `StopReason.BEHAVIOR_STOP`, normal `AgentFinished`
and Session `run_finished` records. This is a safe Continue boundary. Hook
exceptions become `RUNTIME_ERROR`; abort interrupts a slow Hook, and
`wait_for_idle()` includes Hook backpressure. Hooks are not invoked after a Turn
that already produced a terminal result.

### Parallel read-only tools

Parallel Tool Calls are opt-in at both the tool and Agent levels. A handler
must explicitly declare a tool `READ_ONLY`, and the Runtime Policy must enable
parallel calls:

```python
from ejagent import (
    MethodToolHandler,
    RuntimePolicy,
    ToolDefinition,
    ToolEffect,
)

LOOKUP_TOOL = ToolDefinition(
    name="lookup",
    description="Look up one value.",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    effect=ToolEffect.READ_ONLY,
)


class LookupHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((LOOKUP_TOOL,))


agent = BaseAgent(
    model,
    agent_id="parallel-lookups",
    handlers=[LookupHandler()],
    runtime_policy=RuntimePolicy(
        parallel_tool_calls=True,
        max_parallel_tool_calls=4,
    ),
)
```

The scheduler only parallelizes contiguous read-only calls. Unannotated,
unknown, and `SIDE_EFFECTING` tools remain sequential and form barriers between
read batches. The default policy is fully sequential, so existing handlers do
not change behavior after upgrading.

Within a parallel batch, `ToolStarted` is emitted in source order and Progress
may interleave by `tool_call_id`. Handlers execute concurrently, but
`ToolCompleted`, Agent history, and Session Tool Messages are committed in the
Assistant Tool Call order. Errors remain ordinary Tool Results. External abort
settles every active and pending call; if a read-only batch returns a terminal
Tool Control, active peers settle before the first terminal decision in source
order stops the Run. Declaring `READ_ONLY` is a concurrency-safety contract for
the handler and its middleware, not an automatic side-effect inference.

### Streaming responses

`BaseAgent.run()` still returns one final `AgentRunResult`, while provisional
text and provisional reasoning are observed through typed Delta events:

```python
from ejagent import AssistantThinkingDelta, AssistantTextDelta


class ConsoleSink:
    async def emit(self, event):
        if isinstance(event.payload, AssistantThinkingDelta):
            print("[thinking]", event.payload.delta, end="")
        elif isinstance(event.payload, AssistantTextDelta):
            print(event.payload.delta, end="", flush=True)
```

`OpenAIModelAdapter` uses a real streaming request. Tool-call fragments are
assembled inside the provider adapter and only complete `AssistantMessage`
objects enter Agent state. Thinking Delta remains observation-only and is not
mixed into normal text or persisted to Session. Existing complete-only adapters
remain compatible through the default `ModelAdapter.stream()` implementation.
Session recording ignores provisional deltas and persists only
`MessageCompleted`.

### Usage and run budgets

`ModelResponseCompleted` carries optional provider-neutral `ModelUsage`.
Reported Usage is attached to internal agent messages and Session history, but
`AgentContextBuilder` removes it from the final `llm_messages` sent to the
Provider. `AgentRunResult.usage` aggregates all attempted requests while
preserving whether every request actually reported Usage:

```python
result = await agent.run(task="Inspect the project.")

print(result.usage.total_tokens)
print(result.usage.request_count)
print(result.usage.complete)
```

Unknown Usage is distinct from zero. Complete-only adapters remain compatible
and produce an incomplete `RunUsage` unless they override `stream()` with a
terminal Usage value.

### Context pressure and compaction preparation

Context window capacity is independent of cumulative run spend. Configure an
optional `CompactionPolicy` to assess the complete provider request before each
model call:

```python
from ejagent import CompactionPolicy, ContextBudget

context_policy = CompactionPolicy(
    ContextBudget(
        context_window=128_000,
        reserve_tokens=16_000,
        keep_recent_tokens=20_000,
    )
)

agent = BaseAgent(
    model,
    agent_id="context-aware",
    compaction_policy=context_policy,
)
```

The estimate combines the latest assistant `ModelUsage`, trailing messages,
and a UTF-8-aware heuristic lower bound that includes current tool schemas.
Each configured turn emits `ContextPressureEvaluated`. When the threshold is
reached, its `CompactionPreparation` separates protected messages, complete
old User/Assistant/Tool turns to summarize, and recent turns to keep. Tool
calls and results remain in the same turn.

`CompactionPolicy` alone remains observation-only. Applications can call
`estimate_context_usage()` and `prepare_compaction()` directly, and can replace
the fallback through `MessageTokenEstimator`.

### Automatic compaction and overflow recovery

Automatic behavior is opt-in and reuses the same `CompactionPolicy` and
`Compactor`:

```python
from ejagent import AutoCompactionPolicy

agent = BaseAgent(
    model,
    agent_id="context-aware",
    compaction_policy=context_policy,
    compactor=my_compactor,
    auto_compaction_policy=AutoCompactionPolicy(),
)
```

At the configured pressure threshold, Core compacts old complete turns,
rebuilds context, and dispatches the model request in the same Agent Run. If a
provider adapter raises `ContextOverflowError`, Core can compact, rebuild, and
retry once. A second overflow returns `StopReason.CONTEXT_OVERFLOW`; compactor
failure returns `StopReason.COMPACTION_FAILED`. Core never retries after text or
thinking deltas have been exposed, preventing duplicate provisional output.

`AutoCompactionPolicy(compact_on_pressure=False)` keeps overflow recovery while
disabling proactive compaction. Set `enabled=False` or omit the policy to keep
all automatic behavior off. Provider adapters normalize overflow, rate-limit,
timeout, authentication, and other failures through `ModelProviderError` and
`ModelErrorKind`.

### Explicit compaction

A derived agent supplies the summary behavior through the cancellable
`Compactor` protocol, then invokes `compact()` explicitly:

```python
agent = BaseAgent(
    model,
    agent_id="context-aware",
    compaction_policy=context_policy,
    compactor=my_compactor,
)

compaction = await agent.compact()
print(compaction.status)
print(compaction.summary)
```

`ModelCompactor` adapts a borrowed `ModelAdapter` into this protocol while the
application still owns the summary prompt:

```python
compactor = ModelCompactor(
    summary_model,
    context_builder=build_summary_context,
    source="summary-model:v1",
)
```

The injected builder receives `CompactionRequest` and returns the complete
`ContextBuildResult`. The caller owns the borrowed model lifecycle, so Core
does not silently create another provider client or choose a prompt.

The Core calls the Compactor with `CompactionRequest`, creates trusted range and
token metadata in `SummaryEntry`, then atomically installs protected messages +
Summary + recent turns. Failure or cancellation returns a structured
`CompactionResult` and leaves history unchanged. Repeated compaction passes the
previous Summary to the Compactor for merging and replaces the old Summary
message.

`CompactionStarted`, `CompactionCompleted`, and `CompactionFailed` expose the
lifecycle. `abort()` and `wait_for_idle()` apply to compaction as well as normal
runs. `SessionRecorder` stores a compacted recovery snapshot while retaining
the original `SessionMessage` audit entries. Each operation exposes a stable
`operation_id` and `CompactionTrigger`. The Core does not choose a summary model
or prompt.

## Durable Session journals

`SessionRecorder` can use `JsonlSessionStorage` to append a versioned semantic
record for each accepted lifecycle mutation:

```python
from ejagent import JsonlSessionStorage, SessionRecorder

storage = JsonlSessionStorage("./sessions")
recorder = SessionRecorder(session_id="project-42", storage=storage)
agent = BaseAgent(model, agent_id="core-agent", event_sink=recorder)
await agent.run(task="remember this decision")
```

A different process can load the completed snapshot and explicitly restore a
new Agent:

```python
saved = await storage.load("project-42")
if saved is not None:
    resumed = BaseAgent(model, agent_id="core-agent", event_sink=recorder)
    resumed.restore_session(saved)
```

Each JSONL record carries a monotonic `revision`, immutable `record_id`,
`parent_id`, and `branch_id`. File order defines the global revision while
parent links define the logical tree. `SessionRecorder` appends compact mutations such as
`run_started`, `message_appended`, `compaction_applied`, and `run_finished`;
explicit `save()` appends a full Checkpoint for imports and exports.

Branches retain their source history without copying or rewriting records:

```python
forked = await storage.fork("project-42", branch_id="experiment")
rolled_back = await storage.rollback(
    "project-42",
    to_record_id="a-completed-ancestor-record",
    branch_id="rollback-before-change",
)
retry = await storage.prepare_retry(
    "project-42",
    run_id="run-to-repeat",
    branch_id="retry-run",
)
```

`fork()` creates a general branch at a completed projection. `rollback()`
requires the target to be an ancestor of the source head. `prepare_retry()`
branches immediately before a Run and returns its original task; it never
executes that task automatically because Tool calls may have external side
effects. Use `checkout()`, `head()`, and `list_branches()` to inspect the tree.
To continue a branch, restore the checkout and give `SessionRecorder` the same
`branch_id`.

Session IDs are mapped to hashed filenames. Each complete line is encoded
before one append write and followed by `fsync`; an incomplete final line from
an interrupted write is ignored and repaired before the next append. Invalid
JSON in a completed line and unsupported journal schema versions raise
`SessionSerializationError` instead of looking like a missing Session.

`JsonlSessionStorage` coordinates every read-validate-append transaction with a
stable `.jsonl.lock` sidecar. The lock combines process-local coordination with
a POSIX advisory file lock, so separate storage instances and Python processes
cannot allocate the same revision or silently replace a branch head. Conditional
appends still report stale heads as `SessionConflictError`; lock acquisition
timeouts report `SessionLockTimeoutError`. Configure the deadline with
`JsonlSessionStorage(root, lock_timeout=...)`, or pass `None` to wait until the
operation is cancelled. Lock waits run outside the event loop.

`restore_session()` verifies Agent identity and rejects unfinished Runs. Core
does not replay an interrupted Tool call because it may already have produced
an external side effect. The file-backed lock contract targets local POSIX
filesystems; network filesystem deployments must verify their advisory-lock
semantics or provide another `SessionTreeStorage` backend.

## Runtime policy

Tool availability and completion policy are independent:

```python
from ejagent import RuntimePolicy

policy = RuntimePolicy(
    max_steps=20,
    max_no_tool_responses=3,
    max_repeated_tool_calls=3,
    max_run_tokens=None,
    require_explicit_finish=False,
    parallel_tool_calls=False,
    max_parallel_tool_calls=None,
)
```

`max_run_tokens` is an optional cumulative model-request budget. It is checked
between turns: the current response and its requested tools settle first, then
the guard prevents another Provider request with
`StopReason.TOKEN_BUDGET_EXCEEDED`. If another request is needed but Usage was
not reported, the run stops with `StopReason.USAGE_UNAVAILABLE` instead of
treating unknown Usage as zero.

By default, an agent may call tools and later complete with ordinary text. A
derived autonomous agent can require a completion tool:

```python
policy = RuntimePolicy(require_explicit_finish=True)
```

That agent must register one of its own tools that returns
`ToolControl.COMPLETE`.

## Custom tools

Tools are grouped into handlers. `MethodToolHandler` maps a tool named `add` to
an async `do_add()` method:

```python
from collections.abc import Mapping
from typing import Any

from ejagent import (
    CancellationToken,
    MethodToolHandler,
    StepOutcome,
    ToolDefinition,
    ToolEffect,
)

ADD_TOOL = ToolDefinition(
    name="add",
    description="Add two numbers.",
    parameters={
        "type": "object",
        "properties": {
            "left": {"type": "number"},
            "right": {"type": "number"},
        },
        "required": ["left", "right"],
        "additionalProperties": False,
    },
    effect=ToolEffect.READ_ONLY,
    strict=True,
)


class MathHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((ADD_TOOL,))

    async def do_add(
        self,
        arguments: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> StepOutcome:
        return StepOutcome(
            {"value": arguments["left"] + arguments["right"]}
        )
```

Register it explicitly:

```python
agent = BaseAgent(
    OpenAIModelAdapter(ModelConfig.from_env()),
    agent_id="calculator",
    handlers=[MathHandler()],
)
```

Duplicate tool names fail during startup instead of being silently
overwritten. `ToolDefinition` is the Core source of truth for OpenAI function
name, description, Parameters, optional `strict`, and execution Effect.
`to_openai_tool()` returns the provider request shape without exposing Core-only
Effect metadata.

Existing OpenAI function-calling dictionaries remain accepted and are
normalized once when the Handler is created. The compatibility form is not
deprecated and does not require an immediate migration:

```python
MethodToolHandler((OPENAI_TOOL_DICTIONARY,))
```

### Tool progress

Long-running tools can optionally accept a scoped `progress` reporter. Existing
`do_*` methods that do not declare this keyword remain compatible:

```python
from ejagent import ToolProgressReporter, ToolProgressUpdate


async def do_index(
    self,
    arguments,
    *,
    cancellation,
    progress: ToolProgressReporter | None = None,
) -> StepOutcome:
    if progress is not None:
        await progress.report(
            ToolProgressUpdate(
                "indexing files",
                {"completed": 12, "total": 40},
            )
        )
    return StepOutcome({"indexed": 40})
```

Each accepted update becomes a `ToolProgressed` event correlated with the
current run, turn, and tool call. Updates are ordered, stop after cancellation,
and are ignored after `ToolCompleted`. They never change `StepOutcome` or
`ToolControl`, and are not persisted to Agent state or Session.

### Tool control signals

Tool payload and runtime control are separate:

```python
from ejagent import StepOutcome, ToolControl

StepOutcome(data)  # continue the provider-tool loop
StepOutcome(data, control=ToolControl.COMPLETE)
StepOutcome(data, control=ToolControl.REJECT)
StepOutcome(data, control=ToolControl.CANCEL)
```

This lets the runtime distinguish successful completion, policy rejection,
and tool-requested cancellation. `ToolControl.CANCEL` is a tool's business
decision; external `agent.abort()` uses the separate run cancellation
protocol.

## Tool middleware

`ToolMiddleware` is the single interception chain around tool execution:

```python
from ejagent import ToolMiddleware


class AuditMiddleware(ToolMiddleware):
    async def __call__(self, context, call_next):
        print("before", context.tool_name)
        result = await call_next(context)
        print("after", context.tool_name)
        return result
```

The optional `ToolPolicyMiddleware` is one concrete middleware built on that
same chain. It does not add a second policy path or a special `BaseAgent`
parameter:

```python
from ejagent import (
    RuleBasedToolPolicy,
    ToolApprovalDecision,
    ToolEffect,
    ToolPolicyAction,
    ToolPolicyMiddleware,
    ToolPolicyRule,
)


class ConsoleApprover:
    async def approve(self, request):
        # A real application can bridge this request to its UI or RPC layer.
        return ToolApprovalDecision(approved=False, reason="operator denied")


policy = RuleBasedToolPolicy(
    (
        ToolPolicyRule(
            rule_id="approve-side-effects",
            action=ToolPolicyAction.REQUIRE_APPROVAL,
            effects=frozenset({ToolEffect.SIDE_EFFECTING}),
            max_calls_per_run=5,
            reason="this tool can change external state",
        ),
        ToolPolicyRule(
            rule_id="allow-reads",
            action=ToolPolicyAction.ALLOW,
            effects=frozenset({ToolEffect.READ_ONLY}),
            max_calls_per_run=20,
        ),
    ),
    default_action=ToolPolicyAction.DENY,
)

agent = BaseAgent(
    model,
    agent_id="policy-agent",
    handlers=[handler],
    middlewares=[
        ToolPolicyMiddleware(policy, approver=ConsoleApprover()),
        AuditMiddleware(),
    ],
)
```

Rules use ordered, first-match semantics and may select exact tool names,
`ToolEffect`, and a synchronous or asynchronous `when(context)` predicate.
`max_calls_per_run` reserves attempts atomically, including parallel read-only
calls, and resets at each Agent Run start. Approval is fail-closed: a missing
approver, an exception, an invalid decision, or a denied request returns
`ToolControl.REJECT` without invoking the handler. Rejection payloads include
the tool, safe reason, and matching `rule_id`, but never echo arguments.

Core supplies the policy/approval protocol, not an approval UI or
shell/filesystem-specific risk rules. Those remain application concerns.

`ToolSchemaValidationMiddleware` can be placed before policy so structurally
invalid model arguments never consume policy limits, request approval, or
reach a handler:

```python
from ejagent import (
    ToolPolicyMiddleware,
    ToolSchemaValidationMiddleware,
)

agent = BaseAgent(
    model,
    agent_id="validated-agent",
    handlers=[handler],
    middlewares=[
        ToolSchemaValidationMiddleware(max_errors=8),
        ToolPolicyMiddleware(policy, approver=approver),
        AuditMiddleware(),
    ],
)
```

The middleware compiles each canonical `ToolDefinition.parameters` schema
during Agent startup. An invalid registered schema raises
`ToolSchemaConfigurationError` and rolls back startup. A model argument
failure instead becomes a normal, non-terminal Tool Result:

```json
{
  "status": "error",
  "tool": "transfer",
  "code": "invalid_tool_arguments",
  "errors": [
    {
      "path": "/amount",
      "keyword": "type",
      "message": "value does not match the required type"
    }
  ]
}
```

Error paths use JSON Pointer. Messages describe the failed rule without
echoing argument values, and `max_errors` bounds the payload. Tools without a
parameters schema pass through unchanged. Validation results use
`ToolControl.CONTINUE`, so the model can correct its call; permission denial
remains the distinct terminal `ToolControl.REJECT` path.

## MCP tools

MCP uses the same handler contract:

```python
from ejagent import (
    BaseAgent,
    McpToolHandler,
    ModelConfig,
    OpenAIModelAdapter,
)

agent = BaseAgent(
    OpenAIModelAdapter(ModelConfig.from_env()),
    agent_id="browser",
    handlers=[McpToolHandler("examples/mcp_config.json")],
)
```

An MCP-enabled agent can execute MCP tools and then complete with plain text.
It does not need a separate finish tool unless its `RuntimePolicy` explicitly
requires one.

## Skills

Skills are prompt and resource extensions independent of handler tools:

```python
from pathlib import Path

from ejagent import BaseAgent, ModelConfig, OpenAIModelAdapter

agent = BaseAgent(
    OpenAIModelAdapter(ModelConfig.from_env()),
    agent_id="skilled-agent",
    skills_dir=Path("examples/skills"),
)
```

`SkillManager` discovers child folders containing `SKILL.md` and injects compact
metadata containing each skill's name, description, and file location. Users
can explicitly select a skill with `$skill_name` or `skill:skill_name`, which
injects its full instructions into the current context. The core does not
register a special skill tool; a derived agent with a file-reading tool can use
the advertised location for progressive loading.

```text
examples/skills/
  release_notes/
    SKILL.md
    template.md
    examples/
      sample.md
```

## Core boundary

EJAgent Core owns mechanisms:

```text
Orchestration + State + Context + Runtime Policy + Run Result
+ Model Adapter + Tool Protocol + Middleware + MCP + Skills
+ Lifecycle Events + Session Tree + Runtime Cancellation
+ Provider Streaming + Tool Progress + Usage Accounting + Run Budget
+ Context Pressure + Compaction Preparation
+ Model Compactor + Summary Entry + Durable Session Journal
+ Canonical Tool Definition + OpenAI Schema Compatibility View
```

Derived agents own concrete capabilities and policies:

```text
Shell + Filesystem + Git + Workspace + Approval UI
+ Sandbox + Completion Tool + Product Interface
```

See [the Pi Harness comparison](docs/pi-harness-gap-analysis.md) for the
architecture analysis and future roadmap.

## Examples

```bash
# Provider-backed examples
uv run python examples/01_stateful_chat.py
uv run python examples/02_custom_tool.py
uv run python examples/04_mcp_tools.py
uv run python examples/06_skill.py

# Harness examples using the configured real provider
uv run python examples/07_event_observers.py
uv run python examples/08_session_resume.py
uv run python examples/09_runtime_control.py
uv run python examples/10_composed_harness.py
uv run python examples/11_streaming_events.py
uv run python examples/12_tool_progress.py
uv run python examples/13_usage_budget.py
uv run python examples/14_context_pressure.py
uv run python examples/15_explicit_compaction.py
uv run python examples/16_durable_session.py record
uv run python examples/16_durable_session.py resume
```

See [the examples guide](examples/README.md) for the capability demonstrated by
each file.

## Tests

```bash
uv run python -m unittest discover -s tests -p 'test*.py' -q
```

Run the complete local quality gate before submitting a change:

```bash
uv sync --locked --all-extras --group dev
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy
uv build
```

## Release

`.github/workflows/release.yml` builds one verified wheel/sdist pair, attaches
both files to a generated GitHub Release, and publishes the same distributions
to PyPI through Trusted Publishing. No long-lived API token is stored in
GitHub. After configuring the `pypi` environment and PyPI publisher, merge the
release commit into `main`, then push a version-matching tag:

```text
PyPI project: ejagent-core
GitHub owner: jyh20030112
Repository: EJAgent
Workflow: release.yml
Environment: pypi
```

Because `ejagent-core` is a new PyPI project name, configure its Trusted
Publisher before creating the `v0.6.0` release tag.

Protect the GitHub `pypi` environment with required reviewers and restrict
creation of `v*` tags to maintainers. Then publish with:

```bash
git tag v0.6.0
git push origin v0.6.0
```

The workflow rejects tags whose commit is not on `main` or whose value does not
match `project.version`, reruns the complete quality matrix, builds and smoke
tests the distributions, then publishes them to GitHub Releases and PyPI. The
two publishing jobs consume the same immutable Actions artifact; only the PyPI
job receives a short-lived OIDC identity.

## Public API

The package root exports:

- Agent: `BaseAgent`, `AgentOrchestrator`, `AgentState`, `AgentStatus`
- Providers: `ModelAdapter`, `OpenAIModelAdapter`, `ModelConfig`, `AssistantMessage`, `ModelToolCall`, `ModelUsage`, `ModelStreamEvent`, `ModelTextDelta`, `ModelThinkingDelta`, `ModelResponseCompleted`, `ModelErrorKind`, `ModelProviderError`, `ContextOverflowError`, `ModelRateLimitError`, `ModelTimeoutError`, `ModelAuthenticationError`
- Runtime: `RuntimePolicy`, `AgentRunResult`, `RunUsage`, `AgentRunError`, `RunStatus`, `StopReason`
- Behavior: `BehaviorHook`, `BehaviorAction`, `BehaviorDecision`, `BehaviorHookError`, `TurnSnapshot`
- Cancellation: `CancellationToken`, `CancellationSource`, `AgentCancelledError`
- Events: `AgentEvent`, `AgentEventSink`, `CompositeAgentEventSink`, `AgentContinued`, `AssistantTextDelta`, `AssistantThinkingDelta`, `ToolProgressed`, `SteeringApplied`, `SteeringDiscarded`, `ContextPressureEvaluated`, `CompactionStarted`, `CompactionCompleted`, `CompactionFailed`
- Control: `ControlInputKind`, `ControlStatus`, `ControlInput`, `ControlReceipt`, `ContinueRejectedReason`, `ContinueRejectedError`, `FollowUpFailurePolicy`, `FollowUpDiscardReason`, `FollowUpHandle`, `FollowUpError`, `FollowUpRejectedError`, `FollowUpDiscardedError`
- Session: `AgentSession`, `SessionRun`, `SessionRunIntent`, `SessionRecorder`, `SessionStorage`, `SessionJournalStorage`, `SessionTreeStorage`, `MemorySessionStorage`, `JsonlSessionStorage`, `SessionCompaction`, `SessionRecord`, `SessionRecordDraft`, `SessionRecordKind`, `SessionBranchIntent`, `SessionBranch`, `SessionCheckout`, `SessionRetry`, `DEFAULT_SESSION_BRANCH`, `SESSION_SCHEMA_VERSION`, `SESSION_JOURNAL_SCHEMA_VERSION`, `session_to_dict`, `session_from_dict`, `SessionError`, `SessionSerializationError`, `SessionStorageError`, `SessionConflictError`, `SessionLockTimeoutError`
- Context: `AgentContextBuilder`, `ContextBuildResult`, `ContextBudget`, `ContextUsageEstimate`, `CompactionPolicy`, `AutoCompactionPolicy`, `CompactionDecision`, `CompactionPreparation`, `MessageTokenEstimator`, `estimate_context_usage`, `prepare_compaction`
- Compaction: `CompactionRuntime`, `Compactor`, `ModelCompactor`, `CompactionContextBuilder`, `CompactorOutput`, `CompactionRequest`, `CompactionResult`, `CompactionStatus`, `CompactionTrigger`, `SummaryEntry`
- Tools: `ToolDefinition`, `ToolDefinitionError`, `ToolEffect`, `StepOutcome`, `ToolControl`, `ToolProgressUpdate`, `ToolProgressReporter`, `BaseHandler`, `MethodToolHandler`, `McpToolHandler`
- Middleware: `Middleware`, `ToolMiddleware`, `ToolCallContext`, `ToolNext`, `ToolExecutionPolicy`, `RuleBasedToolPolicy`, `ToolPolicyRule`, `ToolPolicyPredicate`, `ToolPolicyAction`, `ToolPolicyDecision`, `ToolPolicyMiddleware`, `ToolApprover`, `ToolApprovalRequest`, `ToolApprovalDecision`, `ToolSchemaValidationMiddleware`, `ToolSchemaConfigurationError`
- Extensions: `McpServerManager`, `SkillManager`

## License

MIT
