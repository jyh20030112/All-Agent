<div align="center">
  <img src="assets/ejagent-mascot.svg" width="180" alt="从蛋中孵化的 EJAgent 吉祥物">
  <h1>EJAgent Core</h1>
  <p><em>孵化属于你的 Agent —— 高度可定制、可持久、可控制的 Python 运行时。</em></p>
  <p>
    <a href="https://github.com/jyh20030112/EJAgent/stargazers"><img src="https://img.shields.io/github/stars/jyh20030112/EJAgent" alt="GitHub Stars"></a>
    <a href="https://pypi.org/project/ejagent-core/"><img src="https://img.shields.io/pypi/v/ejagent-core" alt="PyPI 版本"></a>
    <a href="https://github.com/jyh20030112/EJAgent/actions/workflows/ci.yml"><img src="https://github.com/jyh20030112/EJAgent/actions/workflows/ci.yml/badge.svg" alt="CI 状态"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="MIT License"></a>
    <br>
    <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
    <img src="https://img.shields.io/badge/asyncio-native-4B8BBE" alt="原生 asyncio">
    <img src="https://img.shields.io/badge/MCP-ready-7C3AED" alt="支持 MCP">
  </p>
  <p><a href="README.md">English</a> · <strong>简体中文</strong></p>
</div>

---

EJAgent Core 是一个轻量级 Python Agent 运行时，可用于构建能够调用工具、保持对话
状态、在进程重启后恢复并接受实时控制的 Agent。模型、工具、上下文策略、存储和
可观察性都可以独立替换，让运行时适应你的应用，而不是让应用迁就框架。

它可以作为智能助手、工作流 Agent、编程工具、研究 Agent，以及任何需要可靠
模型—工具循环的应用基础。

## 可以用它做什么

- **高度定制的 Agent**：模型 Provider、工具后端、上下文策略、存储层和观察器均可
  独立替换。
- **有状态的智能助手**：在多个任务间保留类型化对话历史，并从最近一次已提交状态
  继续运行。
- **可持久恢复的 Agent**：将 Session 写入追加式 Journal，进程重启后可以恢复。
- **能够使用工具的 Agent**：接入 Python 函数、组合多个工具执行器，或通过统一接口
  连接 MCP 服务。
- **可控的运行时**：取消当前任务、干预下一次模型调用、排队后续任务，并限制 turn
  数或 token 用量。
- **上下文感知的 Agent**：注入本地 Skill、为长对话生成摘要，或者实现自己的上下文
  策略。
- **可观察的系统**：记录结构化结果、故障、token 用量、模型事件和工具活动，同时让
  观察逻辑与执行过程解耦。
- **不绑定 Provider 的应用**：使用 OpenAI-compatible Endpoint、Anthropic，或为其他
  模型 API 实现适配器。

## 为什么选择 EJAgent Core

做出一个 Agent Demo 很容易，让 Agent 随着应用增长仍然保持可预测则更困难。
EJAgent Core 为执行、状态、工具和外部副作用提供清晰边界，同时足够轻量，可以直接
嵌入现有服务、CLI、Worker 或桌面应用。

项目以两个职责集中的组件为核心：`RuntimeKernel` 执行一次模型—工具 Run，
`AgentHarness` 在多次 Run 之间提供持久状态、资源生命周期、运行控制和原子提交。
这些设计细节不会侵入应用代码，但每个集成边界都可以替换。

## 安装

EJAgent Core 需要 Python 3.12 或更高版本。

```bash
uv add ejagent-core
```

按需添加可选集成：

```bash
uv add 'ejagent-core[anthropic]'  # Anthropic
uv add 'ejagent-core[mcp]'        # MCP
```

## 快速开始

配置 OpenAI-compatible Endpoint：

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.example.com/v1
CHAT_MODEL=your-model
```

创建一个有状态 Agent：

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
    initial_messages=(SystemMessage("请准确回答。"),),
)

async with harness:
    await harness.run("记住我的项目是 EJAgent。")
    answer = await harness.run("我的项目是什么？")
    print(answer.result.output)
```

不改变调用方式，就能为同一个 Agent 增加更多能力：

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

## 每个边界都可以定制

| 想要替换的部分       | 扩展接口           |
| -------------------- | ------------------ |
| 模型 Provider 或协议 | `ModelPort`        |
| 本地或远程工具后端   | `ToolExecutor`     |
| 上下文选择与投影策略 | `ContextPipeline`  |
| 长历史摘要方式       | `ContextCompactor` |
| Session 持久化       | `SessionStore`     |
| 日志、追踪或指标     | `RunObserver`      |

这些接口保持精简并且不绑定 Provider。只需实现应用真正需要的部分，再与内置运行时
组合即可。

## 内置能力

- OpenAI-compatible 与 Anthropic 流式模型适配器
- Python 函数工具、组合工具执行器和 MCP 工具
- 本地 Skill 发现与显式 Skill 激活
- 不改写对话历史的派生上下文摘要
- 内存 Session 与持久化 JSONL Session
- 协作式取消、实时 Steering 和 FIFO Follow-up
- 结构化 Audit 与统一的 token 用量统计
- 基于 Revision、支持幂等和跨进程文件锁的 Session 提交

EJAgent Core 专注于单个逻辑 Agent。应用可以在它的外层按需构建多 Agent 编排和
任意时刻暂停/恢复能力

## 文档

- [全功能使用指南](docs/usage-guide.md)：安装、配置和全部内置能力的使用方式。
- [Core 类与运行链路](docs/core-classes-and-runtime-flow.md)：内部模型与完整 Run
  生命周期。
- [Kernel–Harness 设计](docs/runtime-kernel-harness-design.md)：规范性的架构边界与
  不变量。
- [可运行示例](examples/README.md)：聊天、工具、MCP、Skill、恢复和持久 Session
  示例

## 开发

```bash
uv sync --locked --all-extras --group dev
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run mypy
uv run python -m unittest discover -s tests -p 'test*.py' -q
uv build
```
