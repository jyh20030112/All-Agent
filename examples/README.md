# EJAgent Core Examples

Copy `.env.example` to `.env` and configure a Provider.
Run examples from the repository root, for example:

```bash
uv run python examples/01_stateful_chat.py
```

All examples use the `AgentHarness`/`RuntimeKernel` path and make real Provider
requests.

## Core Examples

- `01_stateful_chat.py`: typed Conversation state across two committed Runs.
- `02_custom_tool.py`: a provider-neutral Tool registered through
  `FunctionToolExecutor`.
- `03_anthropic_chat.py`: the same Harness contract over Anthropic Messages.
- `04_mcp_tools.py`: opt-in MCP integration through `McpToolExecutor`.
- `05_skill.py`: disposable skill indexing and explicit instruction projection.
- `06_session_resume.py`: recovery from `MemorySessionStore` in a new Harness.
- `07_durable_session.py`: cross-process recovery from `JsonlSessionStore`.
- `08_harness_guide.py`: custom tools, live controls, observers, and durable
  recovery in one end-to-end walkthrough.

Every `AgentHarness` has one immutable `agent_id`. Tool-enabled Harnesses expose
only their configured `ToolExecutor.definitions`; plain assistant text completes
a Run by default. A function Tool may return
`ToolExecutionResult(..., control=ToolControl.COMPLETE)` to finish explicitly.

The MCP example requires the commands in `mcp_config.json`; the sample starts
Playwright MCP through `npx`. Install optional support first:

```bash
uv sync --extra mcp
```

The Anthropic example requires `uv sync --extra anthropic` plus
`ANTHROPIC_MODEL` and `ANTHROPIC_API_KEY`.

The Skills example loads `examples/skills/release_notes/`. The catalog index and
explicitly selected `SKILL.md`, template, and sample are projected into a
disposable `ContextView`; they are never committed to Conversation.

For durable recovery, run:

```bash
uv run python examples/07_durable_session.py record
uv run python examples/07_durable_session.py resume
```

Conversation recovery and append-only Run Audit are separate Store domains.

The Harness guide walks the full extension surface in three subcommands:

```bash
uv run python examples/08_harness_guide.py record     # custom tool + durable commit
uv run python examples/08_harness_guide.py resume     # recover in a new process
uv run python examples/08_harness_guide.py controls   # steer / follow_up / cancel
```

`record` commits a Run that calls a custom `add` tool; `resume` reloads the same
`JsonlSessionStore` in a fresh Harness and recalls the tool result from committed
Conversation. `controls` injects a `steer()` instruction mid-Run (transient, never
committed), queues an independent `follow_up()` Run, and cancels a long Run to
show that failed or cancelled Runs stay auditable without advancing revision. A
`RunObserver` prints each RunAudit summary after the Store decision.
