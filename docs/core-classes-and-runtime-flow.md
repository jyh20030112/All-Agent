# EJAgent Harness 类、配置与执行链路

本文面向需要维护、扩展或嵌入 EJAgent Harness 的开发者。它描述当前代码的真实行为，
而不是未来路线图。规范性约束仍以
[`runtime-kernel-harness-design.md`](runtime-kernel-harness-design.md) 和类型定义为准。
需要从安装开始逐项使用功能时，请阅读[全功能使用指南](usage-guide.md)。
项目定位与职责全貌见 [Harness 概览](harness-overview.md)。

## 1. 先建立整体心智模型

EJAgent 是围绕 Agent 决策组织状态、上下文、工具、控制和评估的 Harness。当前每个
`AgentHarness` 实例管理一个逻辑 Agent；`RuntimeKernel` 是其中执行单次 Run 的内核：

```text
AgentHarness（长期存在）
├── SessionSnapshot / ConversationSnapshot
├── SessionStore（提交与恢复）
├── ContextPipeline（一次性上下文投影）
├── ModelPort（Provider 适配）
├── ToolExecutor（工具发现与执行）
├── RunObserver（提交后的旁路观察）
├── TrajectoryMonitor（可选检查点采集与评估）
│   └── 宿主 evaluator → Assessment → 下一次 Context
└── RuntimeKernel（一次 Run）
    └── _RunWorkspace（仅本次 Run 可变）
```

最重要的三类数据不能混用：

| 数据域 | 含义 | 是否持久真相 |
| --- | --- | --- |
| `Conversation` | 下一次 Run 可继续使用的类型化消息 | 是 |
| `Audit` | 成功、失败、取消和外部副作用的事实记录 | 是 |
| `ContextView` | 某次模型请求的摘要、Skills、Steering、轨迹反馈等投影 | 否 |

因此，摘要不会改写 Conversation，失败 Run 的过程仍可进入 Audit，而未提交的消息
不会污染后续对话。

## 2. 最小组装方式

```python
from ejagent.contracts import RunLimits, SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.storage import JsonlSessionStore
from ejagent.tools import FunctionToolExecutor

harness = AgentHarness(
    agent_id="assistant",
    model=OpenAIModelPort(ModelConfig.from_env()),
    tools=FunctionToolExecutor(),
    store=JsonlSessionStore(".ejagent-sessions"),
    initial_messages=(SystemMessage("Answer precisely."),),
    limits=RunLimits(max_turns=12, max_tokens=20_000),
)

async with harness:
    outcome = await harness.run("Explain the current architecture.")
    print(outcome.result.status, outcome.result.output)
```

优先使用 `AgentHarness`。只有嵌入方愿意自行启动资源、构造 `RunSpec`、处理提交时，
才应直接调用 `RuntimeKernel`。

顶层 `ejagent` 只导出常用组装类；完整的值对象、枚举、错误和 Protocol 从
`ejagent.contracts` 导入。Provider、Tool、Context 和 Store 的具体 adapter 分别从
`ejagent.providers`、`ejagent.tools`、`ejagent.context`、`ejagent.storage` 导入。

## 3. 一次 Run 的完整链路

1. `AgentHarness.run()` 自动调用 `start()`，并通过 `_run_lock` 串行化 Run。
2. Harness 从当前 `SessionSnapshot` 捕获 revision 和 Conversation。
3. Harness 创建不可变 `RunSpec`、新的 `CancellationSource` 和 `_RunControls`。
4. `RuntimeKernel.run()` 创建私有 `_RunWorkspace`；TASK Run 先追加
   `UserMessage(task)`，CONTINUE Run 不追加用户消息。
5. Kernel 在每个 turn 开始前检查取消和 token budget。
6. Kernel 只在模型调用前的安全点消费 steering，并将其转换为
   `TransientInstruction`。
7. `ContextPipeline.build()` 从 committed、pending、transient 三部分构建
   一次性的 `ContextView`。
8. Kernel 将 Context 和当前工具定义组成 `ModelRequest`，交给 `ModelPort`。
9. Provider adapter 把厂商流归一为 `ModelTextDelta`、
   `ModelThinkingDelta`，最后必须给出一个 `ModelResponseCompleted`。
10. 没有工具调用时，文本响应结束 Run；有多个工具调用时，Kernel 同时执行整个批次，
    再按模型给出的调用顺序写入结果。
11. 每个工具返回 `ToolExecutionResult`。`CONTINUE` 会把结果加入 workspace 并进入
    下一 turn；`COMPLETE`、`REJECT`、`CANCEL` 立即结束 Run。
12. Kernel 返回 `RunOutcome(result, delta, audit_records, failure)`，但不写 Store。
13. Harness 创建 `SessionCommit` 并调用 Store 的 compare-and-commit。
14. 只有 `RunStatus.COMPLETED` 推进 revision 并将 Delta 加入 Conversation；其他状态
    只追加 Audit。
15. Store 决定完成后，Harness 异步通知 observers。observer 的延迟或失败不改变结果。

启用 `trajectory` 后，Kernel 还会在第一个 Context 前、完整工具批次写入工作区后、
接受文本完成回复前采集检查点。宿主 evaluator 提供当前事实和需求／约束判断，分析器
产生进度及循环评估；连接投影管道后，反馈可进入下一次模型 Context。当前完成建议只
进入 Audit，不会强制继续 Run。见[轨迹集成](trajectory-runtime-readiness.md)。

如果持久化失败，Harness 会把结果改写为 `PERSISTENCE_FAILED`，并保持内存 revision
不变。此时 Store 没有可靠写入，因此 persistence failure 本身只能由返回值和 observer
看到，不能声称已经 durable。Provider、Context 或 Tool 的协议错误属于实现缺陷，会
抛异常；预期的运行错误则进入结构化 `RunFailure`。

## 4. `AgentHarness`：应用入口、生命周期与状态所有者

### 构造参数

| 参数 | 作用 |
| --- | --- |
| `agent_id` | Store key 和逻辑 Agent 身份；同一 Store 中必须稳定。 |
| `model` | 一个 `ModelPort` adapter。 |
| `tools` | 一个 `ToolExecutor`；无工具时传空的 `FunctionToolExecutor()`。 |
| `context` | 可选 `ContextPipeline`；省略时 Kernel 使用 identity 投影。 |
| `trajectory` | 可选 `TrajectoryMonitor`；仅传入监控器不会自动注入模型反馈，还需连接 Context 管道。 |
| `initial_messages` | Store 没有已有快照时的初始 Conversation，通常放 system prompt。 |
| `store` | 省略时只保留 Harness 内的临时状态；持久化使用 `JsonlSessionStore`。 |
| `observers` | Store 决策后异步接收 `RunAudit`，不能参与提交。 |
| `resources` | 额外的 `ManagedResource`，由 Harness 一并管理。 |
| `limits` | 默认 `RunLimits`，可在每次 `run()` 时覆盖。 |
| `configuration_revision` | 写入 `RunSpec` 和 Audit 的配置版本标签。 |
| `run_id_factory` | 自定义 Run ID；测试中可注入确定值。 |
| `clock` | Audit 时间源；测试中可注入固定时钟。 |
| `steering_capacity` | 活跃 Run 的 steering FIFO 容量，默认 16。 |
| `follow_up_capacity` | 已接收但未完成的 follow-up 上限，默认 16。 |

### 状态与方法

`HarnessStatus` 依次为 `NEW → STARTING → READY ↔ RUNNING → STOPPING → CLOSED`。
`start()` 按 Store、Model、Tools、Context、Observers、额外资源的顺序启动满足
`ManagedResource` 的对象；失败时反向回滚。`shutdown()` 停止接收 Run、取消当前
Run、丢弃排队 follow-up、等待 observer，然后反向关闭资源。
启动时如果 Store 返回已有 snapshot，它会替换 `initial_messages`；初始消息只负责
创建一个尚未持久化的新 Conversation。
新 Conversation 即使含初始 system prompt，revision 仍从 0 开始；每个成功 Run
将 revision 增加 1。

- `run(task)`：创建 TASK Run。
- `continue_run()`：不追加用户消息，从已提交 Conversation 创建一个新 Run。
- `cancel(reason)`：协作式取消当前 Run；没有活跃 Run 时返回 `False`。
- `steer(content)`：仅活跃 Run 可接收，在下一个模型调用安全点生效，不进入
  Conversation。
- `follow_up(task)`：仅活跃 Run 可接收，当前链结束后按 FIFO 创建独立 Run。
- `revision`、`messages`、`last_result`、`snapshot`：只反映已提交状态。

`FollowUpHandle` 保存即时 `ControlReceipt`，`await handle.wait()` 返回对应
`RunOutcome`。未接收时抛 `FollowUpRejectedError`；已接收但因关闭被丢弃时抛
`FollowUpDiscardedError`。

## 5. `RuntimeKernel`：一次 Model–Tool 事务

`RuntimeKernel(model, tools, context=None, trajectory=None, clock=None,
monotonic_clock=None)` 没有 Session 和资源生命周期。它只接受 `RunSpec`，返回
`RunOutcome`。`trajectory` 是显式启用、仅观测且 fail-open 的可选边界；为空时不
改变原有 Audit 与结果。内部三个关键类均为私有实现：

- `_RunWorkspace`：保存本 Run 的可变消息、Delta、turn、工具调用 ID 和重复调用计数。
- `_UsageAccumulator`：聚合多次模型请求的 token，并记录有多少请求提供了 usage。
- `_AuditTrail`：生成从 1 开始、严格有序的 `AuditRecord`。

`RunLimits` 当前行为：

| 字段 | 当前作用 |
| --- | --- |
| `max_turns=20` | 模型—工具循环最大 turn 数。 |
| `max_tokens=None` | 下一次模型请求前检查累计总 token；缺 usage 时安全终止。 |
| `max_repeated_tool_calls=3` | 相同工具和参数连续出现达到该次数时终止。 |

工具批次始终并发执行，没有串行开关或并发数量配置。结果按模型调用顺序进入
Conversation；Audit 中的完成事件保留真实完成顺序。

当前也没有可注入的 `RunPolicy` 类；预算、终止和提交资格分别固化在
`RuntimeKernel` 与 `SessionCommit` 中。当前实现尚未提供真正的 mid-Run 暂停/恢复
或多 Agent 协调。

## 6. 消息与 JSON 类

所有核心消息都在 `ejagent.contracts`，与 Provider 字典隔离：

| 类 | 作用 |
| --- | --- |
| `SystemMessage` | 稳定指令，可提交到 Conversation。 |
| `UserMessage` | 用户输入；TASK Run 自动创建。 |
| `AssistantMessage` | 模型文本和零个或多个 `ToolCall`。至少有一种内容。 |
| `ToolCall` | `id`、工具名和不可变 JSON 参数。 |
| `ToolResultMessage` | 与 call ID 配对的结果，可标记 `is_error`。 |
| `ContextSummary` | 派生摘要，只允许出现在 ContextView。 |
| `TransientInstruction` | Steering、Skills、轨迹反馈等 Run-local 指令，不提交。 |

`ConversationMessage` 是 `SystemMessage`、`UserMessage`、`AssistantMessage` 和
`ToolResultMessage` 的闭合联合；`ContextMessage` 额外包含摘要和临时指令。
`freeze_json_value/object()` 会验证 JSON 兼容性并递归冻结 list/dict，
`thaw_json_value()` 只在 Provider 或外部系统需要可变结构时复制解冻。

## 7. Run、结果和审计类

| 类或枚举 | 作用 |
| --- | --- |
| `RunIntent` | `TASK` 或 `CONTINUE`。 |
| `RunSpec` | Harness 捕获的不可变 Run 输入：ID、base revision、消息、限制和 metadata。 |
| `RunDelta` | Kernel 建议追加的消息及其 base revision。 |
| `RunResult` | 状态、停止原因、turn、输出和聚合 usage。 |
| `RunFailure` | 失败 phase、稳定 code、可读消息和 `retryable`。 |
| `RunOutcome` | Result、Delta、Audit records 和可选 Failure 的完整组合。 |
| `RunUsage` | 整个 Run 的 token 与 usage 覆盖率。 |
| `AuditRecord` | 带时区时间戳、序号、kind 和 JSON payload 的事实。 |
| `RunAudit` | Store 决策后的 Run 事实，包含是否 committed 和前后 revision。 |

`RunStatus` 包含 `COMPLETED`、`FAILED`、`CANCELLED`、`REJECTED`。
`StopReason` 描述为何停止；`FailureCode` 描述稳定故障类别；`RunPhase` 指出故障发生在
Context、Model、Tool、Control、Commit 等哪个阶段。部分 `StopReason` 是为后续策略
预留，当前 Kernel 不会产生所有枚举值。

常见 Audit kind 按执行顺序包括 `run_started`、`turn_started`、
`steering_applied`、`context_built`、`model_text_delta`、
`model_thinking_delta`、`assistant_message`、`tool_started`、
`tool_completed`、`turn_completed` 和 `run_finished`。Harness 还可能追加
`steering_discarded` 或 `commit_failed`。
启用轨迹监控时还会出现 `trajectory_checkpointed` 或 `trajectory_capture_failed`。

## 8. Model seam 与 Provider adapters

### 稳定契约

- `ModelPort.stream(ModelRequest, cancellation)`：唯一 Provider seam。
- `ModelRequest`：只含类型化 Context messages 和 `ToolDefinition`。
- `ModelTextDelta` / `ModelThinkingDelta`：流式中间事件。
- `ModelResponseCompleted`：唯一合法终止事件，携带 `AssistantMessage` 和可选
  `ModelUsage`。
- `ModelCallError`：超时、限流、认证等预期故障；Kernel 转为 `RunFailure`。
- `ModelProtocolError`：adapter 返回非法流、工具 JSON 或不完整终止事件。

### OpenAI

`OpenAIModelPort` 使用 Chat Completions streaming，将 Core 消息转换为 OpenAI role
和 function tools，重组分片 tool call，并归一化 usage 与常见错误。它实现
`ManagedResource`；由 Harness 启动时创建 `AsyncOpenAI`，关闭时只关闭自己创建的
client。SDK 自动重试被设为 0，避免绕过 Core 的运行策略。

`ModelConfig` 字段与环境变量：

| 字段 | 环境变量 | 默认值/说明 |
| --- | --- | --- |
| `model` | `CHAT_MODEL` | 必填。 |
| `api_key` | `MODEL_API_KEY` | 必填。 |
| `base_url` | `MODEL_URL` | 必填，支持 OpenAI-compatible endpoint。 |
| `timeout` | `LLM_TIMEOUT` | 60 秒。 |
| `temperature` | `LLM_TEMPERATURE` | 0.7。 |
| `include_usage` | `LLM_INCLUDE_USAGE` | `true`；必须是 true/false。 |

### Anthropic

`AnthropicModelPort` 将 system 指令移到独立字段，把消息转换为 text、tool_use、
tool_result 内容块，重组 `input_json_delta`，并把缓存 token 归一进总 input token。
依赖是可选的：`uv sync --extra anthropic`。它同样关闭 SDK 自动重试。

| `AnthropicConfig` 字段 | 环境变量 | 默认值/说明 |
| --- | --- | --- |
| `model` | `ANTHROPIC_MODEL` | 必填。 |
| `api_key` | `ANTHROPIC_API_KEY` | 必填。 |
| `max_tokens` | `ANTHROPIC_MAX_TOKENS` | 4096，必须大于 0。 |
| `base_url` | `ANTHROPIC_BASE_URL` | 可选。 |
| `timeout` | `ANTHROPIC_TIMEOUT` | 60 秒。 |
| `temperature` | `ANTHROPIC_TEMPERATURE` | 0.7，范围 0–1。 |

两个 adapter 都允许注入已有 client；注入后 client 的关闭责任仍属于调用方。
`ModelUsage` 表示单次请求；Kernel 将其累加成 `RunUsage`。当设置 `max_tokens` 时，
`RunUsage.complete` 必须为真，否则 Kernel 不会冒险发起下一次无法计量的请求。

## 9. Context classes 与实现

`ContextRequest` 明确分开 committed messages、当前 Run pending messages、transient
instructions 和 metadata。`ContextView` 必须保留相同 run ID、source revision 和
turn，否则 Kernel 抛 `ContextProtocolError`。

- `ContextPipeline`：构建一次性 ContextView 的 seam。
- `IdentityContextPipeline`：按 committed → pending → transient 原序输出。
- `DerivedCompactionPipeline`：构造参数为 `compactor` 和
  `minimum_messages=20`；保留开头连续的
  SystemMessage；当其余已提交历史达到阈值时，用 `ContextCompactor` 产生
  `ContextSummary`，再拼接 pending 与 transient。原 Conversation 不变。
- `ContextCompactor`：实际摘要算法 seam，返回 `ContextCompactionOutput`。
- `ContextCompactionRequest`：传给 compactor 的不可变历史和 revision 范围。
- `ContextCompactorError` / `ContextBuildError`：可预期的摘要或上下文失败。
- `SkillsContextPipeline(skills_root, base=None)`：装饰另一个 pipeline。启动时发现
  Skills，每次构建时注入目录索引；任务显式包含 `$name` 或 `skill:name` 时注入完整
  Skill 内容。

`SkillsContextPipeline` 必须由 Harness 启动后使用。它可以包裹压缩 pipeline，例如：

```python
context = SkillsContextPipeline(
    "my_skills",
    base=DerivedCompactionPipeline(my_compactor, minimum_messages=30),
)
```

## 10. Tool classes 与执行语义

| 类 | 作用 |
| --- | --- |
| `ToolDefinition` | 名称、描述和 JSON Schema。 |
| `ToolExecutionResult` | JSON 结果、终止 control、用户输出和可选错误。 |
| `ToolControl` | `CONTINUE`、`COMPLETE`、`REJECT`、`CANCEL`。 |
| `ToolExecutor` | definitions + execute 的稳定 seam。 |
| `ToolExecutionError` | 工具基础设施的预期错误。 |
| `ToolProtocolError` | executor 返回错误类型等实现缺陷。 |

`FunctionTool` 将一个 definition 和 async Python 函数绑定；
`FunctionToolExecutor` 负责名称路由。注意：当前 executor 不自动执行 JSON Schema
校验，工具函数仍需验证 `call.arguments`。

`CompositeToolExecutor` 将多个 executor 合成一个命名空间，拒绝重复工具名，并按
顺序启动、反序关闭子资源。

`McpToolExecutor` 接收 `config_path` 或注入的 `McpManager`，二者必须且只能提供一个。
它在启动后把 MCP schema 转成 Core `ToolDefinition`。内部 `McpServerManager` 使用
FastMCP，工具名改为
`service__tool` 防止跨服务碰撞。依赖通过 `uv sync --extra mcp` 安装。
单个 MCP service 连接失败会被记录并跳过，不会回滚其他已经连接成功的 service；
调用方应在启动后检查 `definitions` 是否符合预期。

```json
{
  "playwright": {
    "command": "npx",
    "args": ["@playwright/mcp@latest", "--headless"]
  }
}
```

Kernel 并发执行同一模型响应中的全部 tool calls。所有结果完成后，按调用源顺序写入
Delta；任何调用失败或 Run 取消时，尚未完成的兄弟任务会被取消。

## 11. 配置来源总览

| 配置来源 | 使用者 | 说明 |
| --- | --- | --- |
| `.env` / 环境变量 | 两种 Provider config | `from_env()` 会先调用 `load_dotenv()`。 |
| MCP 配置文件 | `McpToolExecutor` | MCP service 命令、参数和传输配置。 |
| Skills 目录 | `SkillsContextPipeline` | 每个直接子目录以 `SKILL.md` 为入口。 |
| `pyproject.toml` extras | 安装过程 | `anthropic` 和 `mcp` 均为可选依赖。 |

运行环境要求 Python 3.12+。开发环境使用
`uv sync --locked --all-extras --group dev`，会同时安装全部可选 adapter。

## 12. SessionStore、恢复与提交

`SessionStore` 是 `load()` + compare-and-commit `commit()` 的持久化 seam。
`ConversationSnapshot` 是 revision + committed messages。
`SessionSnapshot` 再加 `agent_id` 和最近一次成功提交的 `last_result`。
`SessionCommit` 包含 base Conversation 与 `RunOutcome`，其关键规则是：

- 只有 `COMPLETED` 的 Run 才 `advances_revision`。
- resulting revision 每个成功 Run 只增加 1，而不是按消息数量增加。
- 同一 run ID + 相同 commit 必须幂等；同一 ID + 不同内容必须冲突。
- Store 必须同时比较 revision 和 base messages，防止 stale write。

省略 Store 时，Harness 直接维护进程内 snapshot，不提供跨 Harness 恢复或历史 Audit
查询。需要长期状态时使用 `JsonlSessionStore`。

`JsonlSessionStore` 是 append-only 持久 adapter：

| 参数 | 作用 |
| --- | --- |
| `root` | journal 目录；文件名是 `sha256(agent_id).core.jsonl`。 |
| `lock_timeout=10.0` | 进程内锁和 POSIX 文件锁等待秒数；`None` 表示无限等待。 |

它对每个完整 JSONL record 校验 schema 和连续 sequence；崩溃留下的最后一个不完整
行会被忽略，并在下次追加前截断。`StoreFileLock` 提供进程内线程锁和 POSIX advisory
lock；因此该 adapter 的跨进程保证目前要求 POSIX。

`storage.codec` 是 Store 私有的稳定 JSON 编解码层，不应作为业务扩展 seam。

相关 Store 错误类：`SessionStoreError` 是预期持久化错误的基类；
`SessionConflictError` 表示 CAS 或 run ID 冲突；
`SessionStoreSerializationError` 表示 journal/schema 损坏；
`SessionStoreLockTimeoutError` 表示文件锁超时。`SessionStoreProtocolError` 则由
Harness 用来报告 Store adapter 返回了
与提议 commit 不一致的对象。

## 13. 控制、取消、Observer 与资源

- `CancellationSource` 拥有可变 cancel 操作；向依赖只传只读
  `CancellationToken`。`token.run(awaitable)` 会在取消时终止被等待任务。
- `ControlKind` 区分 steering 与 follow-up；`ControlStatus` 定义即时接收状态。
- `ControlReceipt` 用 `ACCEPTED`、`NOT_RUNNING`、`TOO_LATE`、`QUEUE_FULL`、
  `CLOSED` 表示即时接收结果，而不是最终执行结果。
- `SteeringInput` 只由 Kernel 在安全点消费一次；Run 结束仍未消费的输入会记录为
  `steering_discarded`。
- `RunControlSource` 是 Kernel 消费 Run-local steering 的最小只读 seam。
- `RunObserver.observe(RunAudit)` 是提交后的旁路 seam。异常被隔离，Harness 关闭时
  才统一等待未完成 observer task。
- `ManagedResource.start()/shutdown()` 是结构化生命周期 seam。Harness 按对象身份
  去重，避免同一资源被启动两次。

## 14. 错误模型

调用方通常先看 `outcome.result.status` 和 `outcome.failure`：

| 情况 | 表现 |
| --- | --- |
| Provider 超时、限流、认证失败 | `FAILED` + `RunFailure(phase=MODEL)` |
| Context 压缩失败 | `FAILED` + `phase=CONTEXT` |
| Tool 基础设施失败 | `FAILED` + `phase=TOOL` |
| 用户取消 | `CANCELLED`，不推进 Conversation |
| Tool 返回 REJECT/CANCEL | `REJECTED`/`CANCELLED`，不推进 Conversation |
| Store 提交失败 | 返回 `PERSISTENCE_FAILED`，Harness snapshot 不变 |
| adapter 返回非法类型或非法流 | 抛 `*ProtocolError`，表示代码缺陷 |
| Harness 已关闭后运行 | 抛 `HarnessClosedError` |
| revision、run ID 或 base 不匹配 | Store 抛 `SessionConflictError` |

## 15. 如何扩展

### 新 Provider

实现 `ModelPort.stream()`，把所有厂商对象留在 adapter 内，只输出 Core stream events；
如需连接生命周期，再实现 `ManagedResource`。参考 OpenAI 和 Anthropic 两个独立实现。

### 新工具后端

实现 `ToolExecutor.definitions` 和 `execute()`。定义必须稳定且名称唯一，执行结果必须是
`ToolExecutionResult`。可通过 `CompositeToolExecutor` 与现有工具组合。

### 新上下文策略

实现 `ContextPipeline.build()`。只读取 `ContextRequest`，返回身份字段一致的
`ContextView`；不能改写 Conversation 或 workspace。

### 新 Store

实现 `SessionStore.load/commit`，并满足 CAS、run ID 幂等和失败不伪装成功的语义。
需要查询历史时额外实现 `AuditReader`。

### 新观察器或资源

实现 `RunObserver` 做日志、指标或追踪；实现 `ManagedResource` 让 Harness 管理连接。
Observer 不适合做必须成功的业务写入，因为其失败不会改变 Run。

## 16. Skills、日志和内部实现类

`Skill` 保存发现后的 name、`SKILL.md`、description，以及可选 template/sample 路径。
`SkillCatalog` 扫描 `skills_root` 的直接子目录，读取 YAML frontmatter，缓存索引和完整
文本，并提供显式名称匹配。它负责读取文件，不负责选择复杂语义；当前自动选择只识别
`$skill_name` 和 `skill:skill_name`。

`setup_logger()` 初始化标准日志格式，`get_logger()` 返回命名 logger。它们是基础设施
辅助函数，不参与 Run 结果或 Audit 语义。

以下类属于实现内部，不应被应用代码依赖：

| 内部类 | 作用 |
| --- | --- |
| `_RunWorkspace` | 隔离本 Run 的可变消息和重复工具调用守卫。 |
| `_UsageAccumulator` / `_AuditTrail` | 聚合 usage、生成严格有序 Audit。 |
| `_RunControls` / `_QueuedFollowUp` | steering 和 follow-up 的内存 FIFO。 |
| `_StreamingToolCall` / `_StreamingToolUse` | 重组 Provider 工具参数分片。 |
| `_StoreIndex` | 重放 JSONL 后得到 snapshot、audit 和幂等索引。 |
| `StoreFileLock` / `_ProcessLockEntry` | 同进程与跨进程文件锁实现。 |
| `McpServerManager` / `_McpToolRoute` | MCP 连接、命名空间与调用路由。 |

`ControlProtocolError`、`ContextProtocolError`、`ModelProtocolError` 和
`ToolProtocolError` 都表示 seam 的实现违反约定，原则上应修 adapter；
`RunCancelledError` 是协作式取消在内部穿越 await 链路的信号，不是最终用户结果。

## 17. 当前类的代码位置

| 模块 | 主要内容 |
| --- | --- |
| `src/ejagent/contracts/` | 全部稳定值对象、枚举、错误和 Protocol。 |
| `src/ejagent/kernel/` | 单 Run 循环和私有 workspace。 |
| `src/ejagent/harness/` | 生命周期、控制队列和提交协调。 |
| `src/ejagent/context/` | Identity、DerivedCompaction、Skills pipelines。 |
| `src/ejagent/providers/` | OpenAI、Anthropic adapters 与配置。 |
| `src/ejagent/tools/` | Function、Composite、MCP executors。 |
| `src/ejagent/storage/` | JSONL Store、锁和 codec。 |
| `src/ejagent/skills/` | `Skill` 与 `SkillCatalog`。 |

建议先阅读测试所描述的行为，再进入 Harness 和 Kernel 实现。
