# EJAgent Core

[English](README.md) | [简体中文](README_zh-CN.md)

EJAgent Core is a small, extensible runtime for one logical agent. It separates
one model–tool execution (`RuntimeKernel`) from durable state, lifecycle, and
control (`AgentHarness`). Conversation, Run Audit, and disposable model context
are distinct data domains.

Requires Python 3.12 or newer.

## Install

```bash
pip install ejagent-core
# source checkout
uv sync --locked --all-extras --group dev
```

Configure an OpenAI-compatible endpoint in `.env`:

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.example.com/v1
CHAT_MODEL=your-model
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.7
LLM_INCLUDE_USAGE=true
```

## Quick Start

```python
from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor

harness = AgentHarness(
    agent_id="assistant",
    model=OpenAIModelPort(ModelConfig.from_env()),
    tools=FunctionToolExecutor(),
    initial_messages=(SystemMessage("Answer precisely."),),
)

async with harness:
    first = await harness.run("Remember that my project is EJAgent.")
    second = await harness.run("What is my project?")
    print(first.result.output, second.result.output)
```

The Harness starts resources transactionally, serializes Runs, commits only
accepted outcomes, and shuts resources down in reverse order. The Kernel never
mutates committed state directly.

## Architecture

```text
AgentHarness
  ├─ Conversation snapshot and revision
  ├─ lifecycle, cancellation, steering, follow-ups
  ├─ SessionStore compare-and-commit
  └─ RuntimeKernel
       ├─ ContextPipeline → disposable ContextView
       ├─ ModelPort → normalized stream
       └─ ToolExecutor → normalized Tool result
```

- `Conversation` contains typed messages usable by future Runs.
- `RunAudit` records completed, failed, cancelled, and rejected attempts.
- `ContextView` may contain summaries, Skills, or steering without rewriting
  Conversation.

## Function Tools

```python
from ejagent.contracts import (
    CancellationToken, ToolCall, ToolDefinition,
    ToolExecutionResult, ToolSemantics,
)
from ejagent.tools import FunctionTool, FunctionToolExecutor

definition = ToolDefinition(
    name="add",
    description="Add two numbers.",
    input_schema={"type": "object"},
    semantics=ToolSemantics.read_only(),
)

async def add(call: ToolCall, cancellation: CancellationToken):
    cancellation.raise_if_cancelled()
    return ToolExecutionResult({
        "value": call.arguments["left"] + call.arguments["right"]
    })

tools = FunctionToolExecutor((FunctionTool(definition, add),))
```

Use `CompositeToolExecutor` to combine independent executors. Duplicate Tool
names fail at the composition boundary.

## MCP

MCP is optional:

```bash
uv sync --extra mcp
```

```python
from ejagent.tools import McpToolExecutor

tools = McpToolExecutor("examples/mcp_config.json")
```

MCP metadata is normalized to Core `ToolDefinition` values after startup. Tool
execution uses the same cancellation and result contract as local functions.

## Skills and Context

```python
from ejagent.context import SkillsContextPipeline

context = SkillsContextPipeline("examples/skills")
```

The pipeline discovers child directories containing `SKILL.md`. It projects a
compact index on every model request and full instructions when the latest user
task names `$skill_name` or `skill:skill_name`. Skill text remains transient.

For long histories, wrap a `ContextCompactor` with
`DerivedCompactionPipeline`. Summaries are derived views and never overwrite
the committed Conversation.

## Persistence and Recovery

Use `MemorySessionStore` for process-local state or `JsonlSessionStore` for an
append-only durable journal:

```python
from ejagent.storage import JsonlSessionStore

store = JsonlSessionStore(".ejagent-sessions")
```

Durable commits use revision and message compare-and-swap, idempotent Run IDs,
cross-process locking, `fsync`, and partial-tail recovery. A Store failure
cannot advance Harness state.

To import an old JSONL Session once:

```python
store = JsonlSessionStore(
    ".ejagent-sessions",
    legacy_session_id="old-session-id",
)
```

Migration reads original entries rather than compacted projections. Unsupported
or unfinished legacy data raises `SessionMigrationError` with remediation.

## Runtime Control

- `harness.cancel(reason)` cooperatively cancels the active Run.
- `harness.steer(content)` admits transient input for the next model safe point.
- `harness.follow_up(task)` queues an independent FIFO Run and returns a handle.
- `harness.continue_run()` starts a Run without appending a new user task.

Arbitrary mid-Run pause/resume and multi-agent management are intentionally out
of scope.

## Extension Contracts

Implement narrow protocols from `ejagent.contracts`:

- `ModelPort` for another Provider protocol.
- `ToolExecutor` for another Tool backend.
- `ContextPipeline` or `ContextCompactor` for context policy.
- `SessionStore` for another durable backend.
- `RunObserver` for post-decision observation.

Expected operational failures use typed error contracts. Invalid configuration
and protocol violations raise exceptions.

## Development

```bash
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run mypy
uv run python -m unittest discover -s tests -p 'test*.py' -q
uv build
```

See [the design document](docs/runtime-kernel-harness-design.md) and
[examples](examples/README.md) for the normative boundaries and runnable usage.
