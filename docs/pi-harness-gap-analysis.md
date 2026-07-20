# SimAgentPlg 与 Pi Agent Harness 能力对照

> 更新日期：2026-07-20
>
> SimAgentPlg 基线：`6161281`，项目版本 `0.5.0`
>
> Pi 子模块基线：`f4e9ca74`
>
> 对照范围：`pi/packages/agent`、`pi/packages/coding-agent` 与 SimAgentPlg Core

## 1. 当前结论

SimAgentPlg 已经不只是 Agent Loop。当前实现覆盖了通用 Agent Core 的主要执行机制：

- Provider-neutral 模型边界和 OpenAI-compatible Adapter
- 模型—工具循环、结构化终止结果与 Runtime Policy
- Tool Runtime、Middleware、MCP Adapter 和 Skill Resource
- 只读生命周期事件、Text / Thinking Stream 和 Tool Progress
- Cancellation、`abort()`、`wait_for_idle()` 与终态事件屏障
- Usage 聚合、Run Token Budget 和 Context Pressure
- 显式与自动 Summary Compaction
- Provider Context Overflow 标准化、最多一次 compact-and-retry
- Durable JSONL Session Journal
- Session Tree、Branch Head、Checkout、Fork、Rollback 和整 Run Retry
- Python 3.12 / 3.13 CI、构建 smoke test 与 PyPI Trusted Publishing CD

因此，`BaseAgent` 已经是一个可独立使用的轻量 Harness 组装根，而不再只是
`AgentOrchestrator` 的薄包装。

与 Pi 的主要差距已经从“缺少持久化和自动压缩”转移到以下领域：

1. **持久化并发正确性**：JSONL 暂未协调同一 Session 的多进程并发写入。
2. **行为控制面**：没有 Steering、Follow-up、Continue 和通用行为型 Hook。
3. **工具调度**：Tool Call 仍顺序执行，工具集合主要在构造或启动时确定。
4. **Provider 广度**：只有 OpenAI-compatible Adapter，Canonical Tool Schema 仍偏 OpenAI。
5. **ExecutionEnv / CodeAgent**：文件、Shell、Git、Workspace、Sandbox、Approval 和 UI/RPC
   明确留给派生层，当前仓库尚未实现该层。

当前最重要的判断是：**先把刚完成的 Durable Session Tree 做到并发可靠，再继续扩展
Harness 控制面。** 否则 Steering、后台任务或多进程服务会放大同一 Journal 的竞争问题。

## 2. 当前架构

```text
BaseAgent
  ├── Active Operation / CancellationSource
  ├── ModelAdapter
  │     └── OpenAIModelAdapter
  ├── AgentOrchestrator
  │     ├── RuntimePolicy
  │     ├── AgentContextBuilder
  │     ├── UsageAccumulator
  │     └── AutoCompactionPolicy
  ├── CompactionPolicy / ContextBudget
  ├── CompactionRuntime
  │     └── Compactor / ModelCompactor
  ├── AgentEventEmitter
  │     └── AgentEventSink / CompositeAgentEventSink
  │           └── SessionRecorder(branch_id)
  │                 ├── MemorySessionStorage
  │                 └── JsonlSessionStorage
  │                       └── Session Tree / Branch Heads
  ├── AgentState
  ├── ToolRuntime
  │     ├── BaseHandler / MethodToolHandler
  │     ├── McpToolHandler
  │     └── ToolMiddleware
  └── SkillManager
```

### 2.1 组件职责

| 组件 | 当前职责 |
|---|---|
| `BaseAgent` | 依赖组装、操作串行化、取消、空闲等待、资源生命周期、Session Restore |
| `AgentOrchestrator` | 模型—工具循环、自动压缩、Overflow Recovery、结构化终止 |
| `AgentState` | 当前消息、Turn、任务状态和运行结果 |
| `AgentContextBuilder` | 构造 Agent 投影和 Provider-safe 请求，注入 Skill 与临时控制消息 |
| `RuntimePolicy` | 步数、无工具响应、重复调用、显式完成和 Run Token Budget |
| `CompactionPolicy` | 评估 Context Pressure，并按完整 User Turn 准备安全切分 |
| `AutoCompactionPolicy` | 控制压力触发和最多一次 Overflow Recovery |
| `CompactionRuntime` | 取消、摘要编排、原子替换和 Compaction 终态事件 |
| `ModelCompactor` | 将借用的 `ModelAdapter` 适配为 `Compactor`，Prompt 仍由应用提供 |
| `AgentEventEmitter` | 分配 `run_id` 和事件序号，按顺序发布只读事件 |
| `SessionRecorder` | 将生命周期事件转换为指定 Branch 的语义 Journal Record |
| `JsonlSessionStorage` | 追加 JSONL、校验树、回放任意节点、管理 Branch 和 Head |
| `ToolRuntime` | 工具路由、Middleware、执行、Progress、控制信号和重复调用保护 |
| `ModelAdapter` | Provider Client 生命周期、请求和响应归一化 |
| `SkillManager` | Skill 发现、metadata、显式选择和上下文投影 |

### 2.2 Runtime 主链路

```text
BaseAgent.run(task)
  → create CancellationToken + run_id
  → AgentState.begin_task(task)
  → AgentStarted
  → AgentContextBuilder.build()
  → ContextPressureEvaluated
  → optional pressure-triggered Compaction
  → ModelAdapter.stream()
  → Text / Thinking Delta*
  → ModelResponseCompleted + ModelUsage
  → MessageCompleted
  → ToolRuntime.execute_tool_call()*
  → ToolProgressed* / ToolCompleted
  → AgentRunResult
  → AgentFinished
  → SessionRecorder append + fsync
```

Provider 在输出任何 Delta 前报告 Context Overflow 时，Orchestrator 可以执行一次自动压缩、
重建 Context 并重试。已经开始输出后发生 Overflow 不会重试，以免重复可见输出。

## 3. 已完成的 Core 能力

### 3.1 执行与终止语义

`AgentRunResult`、`RunStatus`、`StopReason` 和 `ToolControl` 已能稳定区分：

- 文本完成和工具显式完成
- 工具拒绝、工具取消和外部取消
- 空响应、步数限制、无工具响应限制和重复工具调用
- Run Token Budget 超限与 Usage 缺失
- Context Overflow、Compaction 失败和一般 Runtime 错误

工具存在与任务是否必须显式完成已经解耦。`BaseAgent.runtime()` 只作为兼容包装，核心
终态接口是 `run() -> AgentRunResult`。

### 3.2 Provider 与 Tool Runtime

Core 只依赖 `ModelAdapter`，当前 `OpenAIModelAdapter` 负责 Streaming、Tool Call 组装、
Usage 和 Provider Error 归一化。只实现 `complete()` 的 Adapter 仍可通过基类回退工作。

所有工具进入统一 `ToolRuntime`：

- Handler 生命周期和确定性路由
- 重复工具名校验和 JSON 参数解析
- Tool Middleware
- 标准 Tool Message 和结构化 Tool Control
- Cancellation 与 Progress
- 重复调用保护
- MCP Tool 适配

当前同一 Assistant Message 中的 Tool Call 按顺序执行；这与 Pi 当前默认并行策略不同。

### 3.3 事件、Streaming 与取消

事件信封包含 `agent_id`、`run_id` 和单调 `sequence`。当前主要事件包括：

```text
AgentStarted
TurnStarted
ContextPressureEvaluated
AssistantThinkingDelta*
AssistantTextDelta*
MessageCompleted
ToolStarted / ToolProgressed* / ToolCompleted
TurnCompleted
CompactionStarted / CompactionCompleted / CompactionFailed
AgentFinished
```

事件是只读观察协议，不允许 Sink 改写行为。Partial Assistant Message 不进入
`AgentState` 或 Session，只有 `MessageCompleted` 才是提交点。Thinking 和 Progress 默认
不进入后续模型上下文。

`CancellationToken` 已传递到 Model、Compactor、Tool Runtime、Middleware、Handler 和
MCP。`abort()` 不等待操作锁，`wait_for_idle()` 覆盖终态 Sink 收尾；取消后同一 Agent
可以复用。

### 3.4 Usage、Context 与 Compaction

当前上下文管理链路已经完整接通：

```text
ModelUsage
  → RunUsage / max_run_tokens
  → ContextUsageEstimate
  → ContextBudget
  → CompactionPolicy.prepare()
  → Compactor / ModelCompactor
  → SummaryEntry
  → atomic AgentState replacement
```

关键语义：

- Run Budget 与单次请求 Context Window 分离。
- Usage 的 unknown 与明确的 zero 分离。
- UTF-8 启发式估算和 Tool Schema 作为完整请求下界。
- 只在完整 User Turn 边界切分，避免拆开 Tool Call 与 Tool Result。
- 显式 `compact()` 与自动压缩共享取消和原子替换机制。
- 自动压力压缩是 opt-in。
- Provider Overflow 最多 compact-and-retry 一次。
- Compactor 失败安全终止，不安装部分 Summary。
- Compaction 通过 Session Record 持久化，原始审计 Entry 不删除。

### 3.5 Durable JSONL Session Tree

`SessionRecorder` 对 Journal Storage 追加语义 Record：

```text
run_started
message_appended
messages_appended
compaction_applied
run_finished
branch_created
checkpoint
```

每条 Record 包含：

```text
record_id + parent_id + branch_id + revision
session_id + agent_id + sequence + type + data
```

文件顺序定义全局 `revision`，`parent_id` 定义逻辑树。当前支持：

- `load()`：兼容地加载 `main` Head
- `checkout()`：加载 Branch Head 或任意 Record
- `head()` / `list_branches()`
- `fork()`：从已完成投影创建通用 Branch
- `rollback()`：只允许回到源 Branch 的祖先并创建新 Branch
- `prepare_retry()`：回到目标 Run 之前并返回原始 Task
- `SessionRecorder(branch_id=...)`：在指定 Branch 继续执行
- `expected_head_id` 与 `SessionConflictError`

Fork、Rollback 和 Retry 都不改写旧历史。Checkout 可以审计未完成节点，但正常 Fork 和
Restore 会拒绝未完成 Run。Retry 是“准备新 Branch + 返回原 Task”，不会自动重放可能
有外部副作用的 Tool。

JSONL 的当前耐久性保证：

- Session ID 映射为哈希文件名
- 每条完整 Record 通过一次追加写入并 `fsync`
- 中断产生的不完整尾行可忽略并在下次追加前修复
- 完整但损坏的 JSON、错误 Schema、父链和 Branch Head 会明确失败

尚未保证的是多个进程对同一 Session 的 read-validate-append 原子性。

### 3.6 交付与兼容性

- Python 3.12 和 3.13 质量矩阵
- Ruff、Mypy、Unit Test、sdist/wheel 和安装后 Public API smoke test
- PyPI Trusted Publishing CD
- Release Tag 与 `project.version` 一致性检查
- Release Commit 必须属于 `main`

CD 属于交付能力，不改变 Core Runtime 语义。

## 4. 与 Pi 的能力矩阵

| Harness 能力 | Pi 当前实现 | SimAgentPlg 当前实现 | 结论 |
|---|---|---|---|
| Agent Loop | `agentLoop` / `Agent` | `AgentOrchestrator` / `BaseAgent` | 已对齐核心能力 |
| 结构化终止 | Assistant stop reason 为主 | `AgentRunResult` + `StopReason` | Sim 更显式 |
| Provider 边界 | 多 Provider / Model Registry | `ModelAdapter`，仅 OpenAI-compatible | 边界已建，广度不足 |
| Context Transform | `transformContext` | `AgentContextBuilder` | 基础对齐 |
| Event Stream | Message Snapshot + Delta | 分型 Delta + 原子 Message Commit | 语义不同，均可用 |
| Event Barrier | `Agent` Subscriber | 顺序 await `AgentEventSink` | 已具备 |
| Tool Middleware / Hook | `beforeToolCall`、`afterToolCall` | `ToolMiddleware` | 部分具备 |
| Stop-after-turn Hook | `shouldStopAfterTurn` | 无通用接口 | 未实现 |
| Steering | Queue + safe turn boundary | 无 | 未实现 |
| Follow-up | Agent outer loop queue | 无 | 未实现 |
| Continue | `continue()` | 只能通过新 Run / Retry Branch | 语义未对齐 |
| Parallel Tool Calls | 默认并行，可强制顺序 | 顺序执行 | 未实现 |
| Dynamic Tool Set | Runtime 可替换 | 构造/启动时为主 | 未实现 |
| Cancellation | AbortSignal | CancellationToken | 已对齐 |
| Tool Progress | `onUpdate` | `ToolProgressReporter` | 已具备 |
| Usage / Budget | Message Usage + Cost | ModelUsage + RunUsage + Run Budget | 基础已具备，无 Cost |
| Context Pressure | Usage + Estimate | Usage + UTF-8 Estimate + Tool 下界 | 已具备 |
| Explicit Compaction | Harness / Extension Hook | `compact()` + `Compactor` | 已具备 |
| Auto Compaction | Coding Agent 已接通 | `AutoCompactionPolicy` | 已具备 |
| Overflow Recovery | Coding Agent Auto Retry | 标准错误 + 最多一次 Retry | 已具备 |
| Durable Session | JSONL | JSONL Semantic Journal | 已具备 |
| Session Tree | Entry parent tree / forked repo | 同文件 Branch Tree | 已具备，模型不同 |
| Fork | Before / at entry | 从完成 Record Fork | 已具备 |
| Rollback | 通过选 Leaf / Fork 表达 | 独立 `rollback()` 意图 | Sim 更显式 |
| Retry | Continue / Fork 组合 | `prepare_retry(run_id)` | 已具备完整 Run Retry |
| Branch Summary | `branch_summary` Entry / Hook | 只有通用 Compaction Summary | 部分具备 |
| Custom Session Entry | Extension Custom Entry | 固定 Session Record Kind | 未实现扩展协议 |
| Labels / Model Change Entry | 已具备 | 无 | 未实现 |
| ExecutionEnv | Coding Agent 集成 | 无 | 未实现 |
| File / Shell / Git Tools | Coding Agent 内置 | 明确不属于 Core | 待派生 Agent |
| Extension / RPC / TUI | Coding Agent 已具备 | 无 | 待产品层 |

## 5. 关键设计差异

### 5.1 只读 Event 与行为型 Hook

Pi 的 Agent 和 Coding Agent 允许 Hook 阻止 Tool、改写结果、停止 Turn、注入资源或修改
Context。SimAgentPlg 当前坚持：

- `AgentEventSink` 只观察，不修改行为。
- Tool 行为改写只通过 `ToolMiddleware`。
- Context 改写通过显式 `AgentContextBuilder`。
- 自动压缩是 Orchestrator 的明确依赖，不通过 Event 回调重入 Agent。

这个边界避免事件观察者意外改变运行，但也意味着尚缺一个独立于 Event 的行为扩展协议。
后续不能简单把可变 Hook 塞进 `AgentEventSink`。

### 5.2 Session Tree 模型

Pi 当前同时存在 Harness Session Repo 和 Coding Agent Session Manager：

- Entry 使用 `id/parentId` 形成树。
- Coding Agent 支持 Compaction、Branch Summary、Custom Entry、Label、Model Change 等类型。
- Repo Fork 可以把选定 Entry 路径复制到新的 JSONL Session，并记录 Parent Session。

SimAgentPlg 使用一个 Session 一个 JSONL 文件，所有 Branch 共享祖先 Record：

```text
main:       r1 → r2 → r3 → r4
                       ↘ branch_created → r5 → r6
```

这种结构避免复制历史，Fork / Rollback / Retry 意图也可直接审计；代价是同一文件的并发
协调和索引更重要。

### 5.3 Retry 与外部副作用

SimAgentPlg 不从 `ToolCompleted` 中间恢复，也不自动执行 Retry：

1. 找到目标 `RUN_STARTED`。
2. 回到该 Run 之前的完成状态。
3. 创建 Retry Branch。
4. 返回原 Task，由调用方显式执行。

这比无条件 Continue 更保守，因为历史 Tool 可能已经发送邮件、付款或写入外部系统。

### 5.4 Compaction 边界

Pi Coding Agent 支持更丰富的 Branch Summary、Extension Compaction 和 mid-turn 相关语义。
SimAgentPlg 当前只在完整 User Turn 边界切分，并让应用拥有 Summary Prompt。它更适合作为
通用 Core，但在超大单 Turn 或复杂分支摘要方面能力较弱。

### 5.5 Tool 执行顺序

Pi 当前默认并行执行允许并行的 Tool Call，同时保证最终 Tool Result 按 Assistant 源顺序
写入 transcript；SimAgentPlg 全部顺序执行。顺序模型更容易保证副作用和事件顺序，但会
降低多个独立只读 Tool 的吞吐。

## 6. 当前技术边界与风险

### 6.1 JSONL 多进程写入仍是正确性缺口

`JsonlSessionStorage` 的 `asyncio.Lock` 只保护单个实例。两个进程或两个未共享锁的 Storage
实例可能同时读取相同 Head，然后都生成下一 Revision。`expected_head_id` 已定义冲突
语义，但 read-validate-append 尚未被跨进程锁包围。

这是当前最高优先级，因为它可能造成 Journal 损坏，而不是单纯缺少便利功能。

### 6.2 Tree API 尚未完全抽象为 Storage 协议

`SessionJournalStorage` 暴露 `checkout()` 和条件 `append()`，但 `fork()`、`rollback()`、
`prepare_retry()`、`head()` 和 `list_branches()` 仍属于 `JsonlSessionStorage` 具体 API。
如果未来增加数据库或远程 Storage，需要先定义独立 `SessionTreeStorage` 协议。

### 6.3 JSONL 读取成本随 Journal 线性增长

每次操作都会验证完整文件并重建索引；任意节点投影还需要沿父链回放。当前 Checkpoint
可以缩短语义重建，但没有持久 Sidecar Index、Checkpoint Policy 或归档策略。长生命周期
服务需要明确性能上限。

### 6.4 Canonical Tool Schema 仍偏 Provider

Handler Tool Definition 仍使用 OpenAI function-calling 字典。第二个真实 Provider Adapter
加入前，应决定：

- 把当前字典正式定义为 Core Canonical Schema，由 Adapter 转换；或
- 引入强类型 `ToolDefinition`，由各 Adapter 序列化。

### 6.5 Provider 广度不足

只有 OpenAI-compatible Adapter。边界虽然存在，但尚未被 Anthropic、Gemini 或本地模型
Adapter 的真实差异验证，例如 Thinking、Cache Usage、Tool Schema 和错误分类。

### 6.6 Event Backpressure 是同步的

事件按顺序 await，保证确定性和终态屏障，但慢 UI 或网络 Sink 会拖慢 Agent。未来应提供
有界缓冲 Sink Adapter，而不是在 Core 中创建无界后台任务。

## 7. 后续建设顺序

### 阶段一：执行内核——已完成

- Agent Loop、Runtime Policy、Structured Result
- Provider Adapter、Tool Runtime、Skill Resource
- Cancellation、Streaming、Progress、Usage Budget

### 阶段二：Context 与 Compaction——已完成基础版

- Context Pressure、Budget 和完整 Turn Preparation
- Explicit Compaction、ModelCompactor
- Auto Compaction 和 Context Overflow Recovery

### 阶段三：Durable Session Tree——已完成基础版

- Versioned Codec 和 Semantic JSONL Journal
- Checkpoint、Crash Tail Recovery 和 Replay
- Branch Head、Checkout、Fork、Rollback、Retry
- Branch-aware SessionRecorder 和 Head Conflict

### 阶段四：Session Journal Hardening——下一阶段

- 跨实例、跨进程 read-validate-append 原子性
- 后端无关的 `SessionTreeStorage` 协议
- Journal Index / Checkpoint Policy
- 多进程竞争和 Crash Recovery 测试

### 阶段五：Harness 行为控制面

- Steering Queue
- Follow-up Queue
- Continue / Resume-safe-point 语义
- 独立于只读 Event 的行为型 Hook
- Queue 与 Session Journal 的持久化边界

### 阶段六：工具调度与 Provider 扩展

- Parallel Tool Calls 与 Side-effect Policy
- Dynamic Tool Set
- Canonical Tool Definition
- 第二个真实 Provider Adapter

### 阶段七：ExecutionEnv 与派生 CodeAgent

```text
CodeAgent
  ├── ExecutionEnv / Workspace
  ├── Read / Write / Edit
  ├── Grep / Find / List
  ├── Bash / Git
  ├── Sandbox / Approval Policy
  ├── Completion Tool（可选策略）
  └── CLI / RPC / TUI Adapter
```

## 8. 下一步任务建议

下一步建议实现 **Session Journal Hardening**，暂不继续增加新的用户可见 Session 操作。

### 8.1 具体实现范围

1. 定义 `SessionTreeStorage` 协议，统一：
   - `checkout()`
   - `head()` / `list_branches()`
   - `fork()` / `rollback()` / `prepare_retry()`
   - 条件 `append()`
2. 为 JSONL 增加跨进程锁，将以下过程放进同一个临界区：

   ```text
   acquire lock
     → read complete records
     → repair incomplete tail
     → validate expected_head_id
     → assign revision / parent_id
     → append + fsync
   release lock
   ```

3. 保留 `SessionConflictError` 作为 CAS 失败，不把竞争错误伪装成序列化损坏。
4. 明确锁文件生命周期、超时、取消和平台差异；锁等待不能阻塞 Event Loop。
5. 增加独立进程竞争测试：
   - 两个 Writer 同时追加同一 Branch
   - 相同 `expected_head_id` 时只有一个成功
   - 不同 Branch 可以保持正确父链
   - Writer 中断后尾行可恢复
   - Lock Holder 异常退出后其他进程可继续
6. 为长 Journal 增加基准测试，再决定是否立即实现 Sidecar Index。Index 必须可从 JSONL
   重建，不能成为第二个事实来源。

### 8.2 验收标准

- 两个独立 Python 进程不能生成相同 Revision 或破坏 Branch Head。
- 条件追加的冲突稳定返回 `SessionConflictError`。
- 进程崩溃后锁可释放，已有完整 Record 不丢失。
- Python 3.12 / 3.13、macOS / Linux 语义一致。
- 现有 `load(session_id)`、单进程 Recorder 和 0.4 JSONL 文件继续兼容。
- Tree 操作可以仅依赖 `SessionTreeStorage` Protocol 编程。

完成这一阶段后，再进入 Steering / Follow-up。届时队列和后台执行即使由多个 Worker
驱动，也不会建立在一个存在并发破坏窗口的 Session Journal 上。
