# EJAgent Core

[English](README.md) | [简体中文](README_zh-CN.md)

EJAgent Core 是面向单个逻辑 Agent 的小型可扩展运行时。它将一次模型—工具执行
（`RuntimeKernel`）与持久状态、生命周期和控制（`AgentHarness`）分离，并明确区分
Conversation、Run Audit 与一次性的模型 Context。

逐类说明、完整配置参考和 Run 调用链请阅读
[Core 类、配置与运行链路](docs/core-classes-and-runtime-flow.md)。
从安装到全部功能的可复制用法请阅读[全功能使用指南](docs/usage-guide.md)。

需要 Python 3.12 或更高版本。

## 安装

```bash
pip install ejagent-core
# 源码仓库
uv sync --locked --all-extras --group dev
```

Anthropic Provider 适配器按需安装：`pip install 'ejagent-core[anthropic]'`。

在 `.env` 中配置 OpenAI-compatible Endpoint：

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.example.com/v1
CHAT_MODEL=your-model
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.7
LLM_INCLUDE_USAGE=true
```

## 快速开始

```python
from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor

harness = AgentHarness(
    agent_id="assistant",
    model=OpenAIModelPort(ModelConfig.from_env()),
    tools=FunctionToolExecutor(),
    initial_messages=(SystemMessage("请准确回答。"),),
)

async with harness:
    first = await harness.run("记住我的项目是 EJAgent。")
    second = await harness.run("我的项目是什么？")
    print(first.result.output, second.result.output)
```

Harness 以事务方式启动资源、串行化 Run、只提交 Store 接受的结果，并按相反顺序
关闭资源。Kernel 永远不会直接修改已提交状态。

## 架构

```text
AgentHarness
  ├─ Conversation 快照与 Revision
  ├─ 生命周期、取消、Steering、Follow-up
  ├─ SessionStore Compare-and-Commit
  └─ RuntimeKernel
       ├─ ContextPipeline → 一次性 ContextView
       ├─ ModelPort → 标准化 Stream
       └─ ToolExecutor → 标准化 Tool Result
```

- `Conversation` 只包含可用于未来 Run 的类型化消息。
- `RunAudit` 记录成功、失败、取消和拒绝的尝试。
- `ContextView` 可以包含摘要、Skill 或 Steering，但不会改写 Conversation。

## Function Tool

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

通过 `CompositeToolExecutor` 可以组合多个独立 Executor；重复 Tool 名称会在组合
边界立即失败。

## MCP

MCP 是可选依赖：

```bash
uv sync --extra mcp
```

```python
from ejagent.tools import McpToolExecutor

tools = McpToolExecutor("examples/mcp_config.json")
```

MCP 元数据会在启动后转换成 Core `ToolDefinition`，执行时复用本地 Tool 相同的取消
与结果协议。

## Skill 与 Context

```python
from ejagent.context import SkillsContextPipeline

context = SkillsContextPipeline("examples/skills")
```

Pipeline 会发现包含 `SKILL.md` 的子目录。每次模型请求都会获得精简索引；最新用户
任务出现 `$skill_name` 或 `skill:skill_name` 时才投影完整说明。Skill 文本不会进入
Conversation。

长历史可以通过 `DerivedCompactionPipeline` 和自定义 `ContextCompactor` 生成摘要；
摘要始终是派生视图，不会覆盖已提交 Conversation。

## 持久化与恢复

`MemorySessionStore` 用于进程内状态，`JsonlSessionStore` 提供追加式持久 Journal：

```python
from ejagent.storage import JsonlSessionStore

store = JsonlSessionStore(".ejagent-sessions")
```

持久提交具备 Revision 和消息 Compare-and-Swap、幂等 Run ID、跨进程锁、`fsync`
和不完整尾部恢复。Store 失败不会推进 Harness 状态。

一次性导入旧 JSONL Session：

```python
store = JsonlSessionStore(
    ".ejagent-sessions",
    legacy_session_id="old-session-id",
)
```

迁移以原始 Entry 为准，不使用旧 Compaction 投影。无法无损表示或尚未完成的数据会
抛出带修复建议的 `SessionMigrationError`。

## 运行控制

- `harness.cancel(reason)`：协作式取消当前 Run。
- `harness.steer(content)`：为下一个模型安全点提交临时输入。
- `harness.follow_up(task)`：排队一个独立的 FIFO Run，并返回 Handle。
- `harness.continue_run()`：不追加新用户任务，直接从已提交 Revision 继续。

任意时刻暂停/恢复和多 Agent 管理明确不在 Core 范围内。

## 扩展协议

通过 `ejagent.contracts` 中的窄协议扩展：

- `ModelPort`：接入其他 Provider 协议。
- `ToolExecutor`：接入其他 Tool Backend。
- `ContextPipeline` / `ContextCompactor`：定义 Context 策略。
- `SessionStore`：接入其他持久化 Backend。
- `RunObserver`：观察 Store 决策后的 Run。

预期的运行故障使用类型化错误；无效配置和协议违规使用异常。

## 开发

```bash
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run mypy
uv run python -m unittest discover -s tests -p 'test*.py' -q
uv build
```

规范边界和可运行示例见[设计文档](docs/runtime-kernel-harness-design.md)与
[示例指南](examples/README.md)。
