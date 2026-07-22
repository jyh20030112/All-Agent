# EJAgent Core Examples

All examples use the environment variables documented in the project README.
Copy `.env.example` to `.env` and fill in credentials for an OpenAI-compatible
provider before running them. Examples `07` through `16` exercise the Harness
against the real `OpenAIModelAdapter`; they do not use scripted model results.

Run an example from the repository root:

```bash
uv run python examples/01_stateful_chat.py
```

Every `BaseAgent` declares its own immutable `agent_id`.

## Examples

- `01_stateful_chat.py`: plain chat, conversation memory, and `reset()`
- `02_custom_tool.py`: an OpenAI-first canonical `ToolDefinition` registered
  through `MethodToolHandler`
- `04_mcp_tools.py`: opt-in MCP integration with a custom config file
- `06_skill.py`: local skill discovery, indexing, template, and sample injection
- `07_event_observers.py`: ordered lifecycle events and multiple independent
  sinks
- `08_session_resume.py`: linear Session recording and recovery in a new agent
- `09_runtime_control.py`: `abort()`, `wait_for_idle()`, terminal events, and reuse
- `10_composed_harness.py`: Model Adapter, tools, Middleware, events, Session,
  runtime policy, and explicit completion composed as one Harness
- `11_streaming_events.py`: real provider Thinking and Text Delta events
  rendered as they arrive while the final message remains atomically committed
- `12_tool_progress.py`: a real provider-triggered tool reports structured,
  ordered progress before its final result is committed
- `13_usage_budget.py`: real Provider Usage is normalized, aggregated, and used
  to stop before an intentionally over-budget follow-up request
- `14_context_pressure.py`: the complete real Provider request is assessed,
  old full turns are prepared for compaction, and current history remains
  unchanged
- `15_explicit_compaction.py`: a real Provider-backed Compactor atomically
  replaces old Tool history with a Summary Entry, then the Agent and Session
  continue from the compacted projection
- `16_durable_session.py`: append a versioned JSONL Session Journal and restore
  it in a separate process invocation
- `17_session_tree.py`: inspect branches and prepare Fork, Rollback, or Retry
  branches without automatically executing a model

Run the composed Harness example directly:

```bash
uv run python examples/10_composed_harness.py
```

The Harness examples make real provider requests, so response text and exact
timing depend on the configured model. Their control flow, event ordering,
Session projection, and terminal result protocols remain deterministic.

Tool-enabled agents expose only the handlers explicitly passed to
`BaseAgent`. Tool availability does not force an explicit completion call;
plain text completes a task by default. A derived agent can require explicit
completion through `RuntimePolicy` and return
`StepOutcome(..., control=ToolControl.COMPLETE)` from its own completion tool.
The repeated-tool-call guard remains configurable through the same policy.
Example `13` defaults `HARNESS_MAX_RUN_TOKENS` to `1` so its first reported
response exhausts the budget after the requested tool settles. Set a larger
value to observe additional turns.
Example `14` uses `HARNESS_CONTEXT_WINDOW`, `HARNESS_CONTEXT_RESERVE`, and
`HARNESS_KEEP_RECENT_TOKENS` only as Harness-side demonstration values. The
default low threshold makes the seeded old tool output produce a compaction
suggestion; it does not change the configured model's real context window.
Example `15` uses the configured Provider twice: once through an example-owned
summary prompt and once for normal Agent continuation. The Core supplies the
protocol and atomic state transition but does not own that prompt.

The MCP example requires the commands declared in `mcp_config.json` to be
available locally. Its sample configuration starts the Playwright MCP server
through `npx`.

The skill example loads `examples/skills/release_notes/`. Skill metadata,
including its file location, is injected into the model context. Naming the
skill explicitly loads its full `SKILL.md`, optional `template.md`, and optional
`examples/sample.md` content. The core does not register a special skill tool.

The event examples use `CompositeAgentEventSink` to attach observers without
changing the Agent Loop. The Session examples use `SessionRecorder` and
`MemorySessionStorage`; only real user, assistant, and tool messages are saved.
Tool Progress is observation-only and is not written to Agent state, Session,
or the model transcript.
The runtime-control example distinguishes external `abort()` from
`ToolControl.CANCEL` and waits until terminal event sinks have settled before
reporting idle. It waits briefly before aborting the provider request; set
`HARNESS_ABORT_DELAY` to adjust that delay.

`16_durable_session.py` appends a versioned JSONL Session Journal. Run it once
with `record` and again with `resume`; the second invocation creates a new Agent
process and restores the saved conversation through `restore_session()`.
`17_session_tree.py` operates on that same journal. Its `branches`, `fork`,
`rollback`, and `retry` commands expose tree management while leaving model
execution explicit.
