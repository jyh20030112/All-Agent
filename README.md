<div align="center">
  <img src="assets/ejagent-mascot.svg" width="180" alt="EJAgent mascot hatching from an egg">
  <h1>EJAgent Core</h1>
  <p><em>Hatch an agent of your own — a highly customizable Python runtime for reliable AI agents.</em></p>
  <p>
    <a href="https://github.com/jyh20030112/EJAgent/stargazers"><img src="https://img.shields.io/github/stars/jyh20030112/EJAgent" alt="GitHub stars"></a>
    <a href="https://pypi.org/project/ejagent-core/"><img src="https://img.shields.io/pypi/v/ejagent-core" alt="PyPI version"></a>
    <a href="https://github.com/jyh20030112/EJAgent/actions/workflows/ci.yml"><img src="https://github.com/jyh20030112/EJAgent/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="MIT license"></a>
    <br>
    <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
    <img src="https://img.shields.io/badge/asyncio-native-4B8BBE" alt="asyncio native">
    <img src="https://img.shields.io/badge/MCP-ready-7C3AED" alt="MCP ready">
  </p>
  <p><strong>English</strong> · <a href="README_zh-CN.md">简体中文</a></p>
</div>

---

EJAgent Core is a lightweight Python runtime for creating agents that can use
tools, retain conversation state, recover after restarts, and accept live
control. Its model, tools, context strategy, storage, and observability are all
replaceable, so you can adapt the runtime to your application instead of
adapting your application to a framework.

Use it as the foundation for assistants, workflow agents, coding tools,
research agents, or any application that needs a reliable model–tool loop

## What You Can Build

- **Highly customized agents** — replace the model provider, tool backend,
  context strategy, storage layer, and observers independently.
- **Stateful assistants** — keep typed conversation history across multiple
  tasks and continue from the latest committed state.
- **Durable agents** — persist sessions to an append-only journal and recover
  them after a process restart.
- **Tool-using agents** — expose Python functions, compose multiple tool
  executors, or connect MCP services through one consistent interface.
- **Controllable runtimes** — cancel active work, steer the next model step,
  queue follow-up tasks, and enforce turn or token limits.
- **Context-aware agents** — inject local Skills, derive summaries for long
  conversations, or implement your own context policy.
- **Observable systems** — capture structured results, failures, token usage,
  model events, and tool activity without coupling observers to execution.
- **Provider-flexible applications** — use OpenAI-compatible endpoints,
  Anthropic, or implement a provider adapter for another model API.

## Why EJAgent Core

Agent demos are easy; agents that remain predictable as an application grows
are harder. EJAgent Core provides explicit boundaries for execution, state,
tools, and side effects while staying small enough to embed in an existing
service, CLI, worker, or desktop application.

At its center are two focused components: `RuntimeKernel` executes one
model–tool Run, while `AgentHarness` adds durable state, resource lifecycle,
runtime control, and atomic commits across Runs. The detailed design stays out
of your application code, but every integration boundary remains replaceable.

## Install

EJAgent Core requires Python 3.12 or newer.

```bash
uv add ejagent-core
```

Add an optional integration when needed:

```bash
uv add 'ejagent-core[anthropic]'  # Anthropic
uv add 'ejagent-core[mcp]'        # MCP
```

## Quick Start

Configure an OpenAI-compatible endpoint:

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.example.com/v1
CHAT_MODEL=your-model
```

Create a stateful agent:

```python
from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor

model = OpenAIModelPort(ModelConfig.from_env())
harness = AgentHarness(
    agent_id="assistant",
    model=model,
    tools=FunctionToolExecutor(),
    initial_messages=(SystemMessage("Answer precisely."),),
)

async with harness:
    await harness.run("Remember that my project is EJAgent.")
    answer = await harness.run("What is my project?")
    print(answer.result.output)
```

The same agent can be upgraded without changing its calling style:

```python
from ejagent.context import SkillsContextPipeline
from ejagent.storage import JsonlSessionStore
from ejagent.tools import McpToolExecutor

harness = AgentHarness(
    agent_id="assistant",
    model=model,
    tools=McpToolExecutor("mcp_config.json"),
    context=SkillsContextPipeline("skills"),
    store=JsonlSessionStore(".ejagent-sessions"),
)
```

## Customize Every Boundary

| You want to change               | Extension point    |
| -------------------------------- | ------------------ |
| Model provider or protocol       | `ModelPort`        |
| Local or remote tool backend     | `ToolExecutor`     |
| Context selection and projection | `ContextPipeline`  |
| Long-history summarization       | `ContextCompactor` |
| Session persistence              | `SessionStore`     |
| Logging, tracing, or metrics     | `RunObserver`      |

These are narrow, provider-neutral contracts. Implement only the part your
application needs, then compose it with the built-in runtime.

## Built-in Capabilities

- OpenAI-compatible and Anthropic streaming model adapters
- Python function tools, composite tool executors, and MCP tools
- Local Skill discovery and explicit Skill activation
- Derived context compaction without rewriting conversation history
- In-memory sessions and durable JSONL sessions
- Cooperative cancellation, live steering, and FIFO follow-ups
- Structured audit records and normalized usage accounting
- Revision-based, idempotent session commits with cross-process file locking

EJAgent Core intentionally focuses on one logical agent. Multi-agent
orchestration and arbitrary mid-Run pause/resume can be built around it when an
application needs them.

## Documentation

- [Full Usage Guide](docs/usage-guide.md) — installation, configuration, and
  recipes for every built-in capability.
- [Core Classes and Runtime Flow](docs/core-classes-and-runtime-flow.md) — the
  internal model and complete Run lifecycle.
- [Kernel–Harness Design](docs/runtime-kernel-harness-design.md) — normative
  architectural boundaries and invariants.
- [Runnable Examples](examples/README.md) — focused examples for chat, tools,
  MCP, Skills, recovery, and durable sessions.

## Development

```bash
uv sync --locked --all-extras --group dev
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run mypy
uv run python -m unittest discover -s tests -p 'test*.py' -q
uv build
```
