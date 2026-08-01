# EJAgent Core Examples

Copy `.env.example` to `.env` and configure an OpenAI-compatible provider.
Run examples from the repository root, for example:

```bash
uv run python examples/01_stateful_chat.py
```

All examples use the `AgentHarness`/`RuntimeKernel` path with
`OpenAIModelPort` and make real Provider requests.

## Core Examples

- `01_stateful_chat.py`: typed Conversation state across two committed Runs.
- `02_custom_tool.py`: a provider-neutral Tool registered through
  `FunctionToolExecutor`.
- `04_mcp_tools.py`: opt-in MCP integration through `McpToolExecutor`.
- `06_skill.py`: disposable skill indexing and explicit instruction projection.
- `08_session_resume.py`: recovery from `MemorySessionStore` in a new Harness.
- `16_durable_session.py`: cross-process recovery from `JsonlSessionStore`.

Every `AgentHarness` has one immutable `agent_id`. Tool-enabled Harnesses expose
only their configured `ToolExecutor.definitions`; plain assistant text completes
a Run by default. A function Tool may return
`ToolExecutionResult(..., control=ToolControl.COMPLETE)` to finish explicitly.

The MCP example requires the commands in `mcp_config.json`; the sample starts
Playwright MCP through `npx`. Install optional support first:

```bash
uv sync --extra mcp
```

The Skills example loads `examples/skills/release_notes/`. The catalog index and
explicitly selected `SKILL.md`, template, and sample are projected into a
disposable `ContextView`; they are never committed to Conversation.

For durable recovery, run:

```bash
uv run python examples/16_durable_session.py record
uv run python examples/16_durable_session.py resume
```

Conversation recovery and append-only Run Audit are separate Store domains.
