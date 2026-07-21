# SimAgentPlg 与 Pi Agent Harness 能力对照

> 更新日期：2026-07-21
>
> SimAgentPlg 基线：`da09e60` 后的 Session、Steering 与 Follow-up 工作树，项目版本 `0.5.0`
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
- 跨实例 / 跨进程 Session Journal 原子写入与 `SessionTreeStorage` 协议
- 有界 FIFO Steering Queue、Model-call Safe Point 与 Session 审计
- Follow-up Run Chain、有界 FIFO、Waitable Handle 与终态 Sink 屏障
- Continue / Resume Safe Point、独立 Run Intent 与 Session 回放
- Python 3.12 / 3.13 CI、构建 smoke test 与 PyPI Trusted Publishing CD

因此，`BaseAgent` 已经是一个可独立使用的轻量 Harness 组装根，而不再只是
`AgentOrchestrator` 的薄包装。

与 Pi 的主要差距已经从“缺少持久化和自动压缩”转移到以下领域：

1. **行为控制面**：Steering、Follow-up 与 Continue 基础版已完成；尚无通用行为控制策略。
2. **工具调度**：Tool Call 仍顺序执行，工具集合主要在构造或启动时确定。
3. **Provider 广度**：只有 OpenAI-compatible Adapter，Canonical Tool Schema 仍偏 OpenAI。
4. **长 Journal 性能**：写入与读取仍会线性重建索引，需要按实际规模决定索引和归档策略。
5. **ExecutionEnv / CodeAgent**：文件、Shell、Git、Workspace、Sandbox、Approval 和 UI/RPC
   明确留给派生层，当前仓库尚未实现该层。

Session Journal 的并发正确性边界、Steering Safe Point、Follow-up Run Chain 和 Continue
Resume Safe Point 已补齐。行为控制面的下一批应实现独立于只读 Event 的 Behavior Hooks。

## 2. 当前架构

```text
BaseAgent
  ├── Active Operation / CancellationSource
  ├── Bounded Steering Queue / Control Receipt
  ├── Follow-up FIFO / Run Handle / Run-chain Gate
  ├── Continue Safe Point / Run Intent
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
| `BaseAgent` | 依赖组装、Run Chain、Steering、Follow-up、取消、空闲等待、资源生命周期、Session Restore |
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
| `JsonlSessionStorage` | 原子追加 JSONL、跨进程锁、校验树、回放节点、管理 Branch 和 Head |
| `ToolRuntime` | 工具路由、Middleware、执行、Progress、控制信号和重复调用保护 |
| `ModelAdapter` | Provider Client 生命周期、请求和响应归一化 |
| `SkillManager` | Skill 发现、metadata、显式选择和上下文投影 |

### 2.2 Runtime 主链路

```text
BaseAgent.run(task)
  → create CancellationToken + run_id
  → AgentState.begin_task(task)
  → AgentStarted
  → drain Steering at model-call safe point
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
  → SessionRecorder locked append + fsync
  → optional next Follow-up Run
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
AgentContinued
TurnStarted
SteeringApplied / SteeringDiscarded
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

### 3.4 Steering 控制面

`BaseAgent.steer(content)` 为当前 Run 提交有界 FIFO 控制输入，并返回带稳定 `input_id` 的
`ControlReceipt`。即时状态明确区分 `accepted`、`agent_idle`、`queue_full` 和
`run_closing`，不会静默丢弃输入。

Steering 不会中断正在执行的 Model Response 或 Tool。Core 在下一次 Provider Context 最终
确定前消费队列；该安全点也覆盖 Context Pressure Compaction 和 Overflow Retry。已经开始的
Tool Call 必须先产生对应 Tool Result，Steering 才会作为新的 User Message 进入上下文。

成功应用会发出 `SteeringApplied` 并写入 `steering_applied` Session Record。Run 在下一个
Model Call 前结束时，未消费输入发出 `SteeringDiscarded`，但不伪装成 Durable History。

### 3.5 Follow-up Run Chain

`BaseAgent.follow_up(task)` 将独立任务提交到当前 Run Chain 的有界 FIFO，并返回
`FollowUpHandle`。Handle 的 Receipt 只表示是否入队；`wait()` 最终返回该独立 Run 的
`AgentRunResult`。取消某个等待者不会取消队列任务。

Follow-up 必须等待前一个 `AgentFinished` 和全部 Event Sink 完成后才开始。每项都有独立
`run_id`、事件序列和 Session Run。额外的 Run-chain Gate 会阻止已经等待锁的直接 `run()`、
`compact()` 或生命周期操作插入 Follow-up 之前；首个直接 `run()` 仍会在自身结果完成后立即
返回，不等待整条 Follow-up Chain。

默认 `FollowUpFailurePolicy.DISCARD` 会在前一 Run 失败或取消时丢弃余项；显式选择
`CONTINUE` 才继续。Queue Full、Agent Idle、Shutdown、前序 Run 非成功和等待者取消都有明确
Receipt 或异常。Pending Queue 只存在于进程内，只有真正开始的 Follow-up 才通过
`SessionRecorder` 写入 `run_started`。

### 3.6 Continue / Resume Safe Point

`BaseAgent.continue_run()` 从已提交历史启动新的独立 Run，但不会追加 User Message。Continue
分配新的 `run_id`，发出 `AgentContinued`，并通过 `run_continued` Journal Record 持久化；
`SessionRunIntent` 可区分普通 Task 与 Continue，Restore 后仍能恢复最新终态并继续。

Continue 只允许显式安全的 Stop Reason。Active Run、Shutdown、无前序 Run、不支持的失败或
取消原因，以及未匹配 Tool Result 的 Tool Call 都通过 `ContinueRejectedReason` 明确拒绝。
调用方可以先读取 `can_continue` 和 `continue_rejection_reason`，也可以处理
`ContinueRejectedError`。

Continue 复用普通 Run 的 Cancellation、Steering Safe Point、Auto Compaction、Follow-up
Run Chain、`AgentFinished` Sink Barrier 和 `wait_for_idle()`。因此它不是修改旧 Run，也不是
Retry Branch；它只是从安全历史边界继续模型循环。

### 3.7 Usage、Context 与 Compaction

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

### 3.8 Durable JSONL Session Tree

`SessionRecorder` 对 Journal Storage 追加语义 Record：

```text
run_started
message_appended
messages_appended
steering_applied
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

### 3.9 交付与兼容性

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
| Steering | Queue + safe turn boundary | 有界 FIFO + Model-call Safe Point | 基础已具备 |
| Follow-up | Agent outer loop queue | 独立 FIFO Run Chain + Waitable Handle | 基础已具备 |
| Continue | `continue()` | 独立 Run + 无新增 User Message + Safe Point | 基础已具备 |
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

### 6.1 JSONL 多进程写入正确性——已加固

`JsonlSessionStorage` 现在为每个 Session 使用稳定 `.jsonl.lock` Sidecar，并组合进程内共享锁
与 POSIX Advisory Lock。读取、修复尾行、校验 `expected_head_id`、分配 Revision / Parent、
追加和 `fsync` 位于同一临界区。CAS 失败继续返回 `SessionConflictError`，锁等待超时返回
`SessionLockTimeoutError`；等待在 Worker Thread 中执行，可以取消而不阻塞 Event Loop。

该实现面向 macOS / Linux 本地文件系统。NFS 等网络文件系统需要验证 Advisory Lock 语义，
或者提供其他 `SessionTreeStorage` Backend。

### 6.2 Tree Storage 协议——已完成

公开的 `SessionTreeStorage` 在 `SessionJournalStorage` 之上统一 `records()`、`head()`、
`list_branches()`、`fork()`、`rollback()` 和 `prepare_retry()`。上层可以仅依赖协议编程，
不必绑定 JSONL 具体实现。

### 6.3 JSONL 读取成本随 Journal 线性增长

每次操作都会验证完整文件并重建索引；任意节点投影还需要沿父链回放。当前 Checkpoint
可以缩短语义重建，但没有持久 Sidecar Index、Checkpoint Policy 或归档策略。长生命周期
服务需要明确性能上限。

可重复基准位于 `benchmarks/session_journal.py`。本机 1,000 条小型 Checkpoint Journal
约 401 KB，完整 Load 约 14.7 ms；从空文件逐条构建约 7.68 s，平均追加约 7.68 ms。
这验证了当前规模仍可用，但也确认逐次追加的累计成本呈二次增长。现阶段不增加 Index；
当真实 Session 接近数千至数万 Record 时，再设计可完全从 JSONL 重建的 Sidecar Index。

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

### 阶段四：Session Journal Hardening——已完成

- 跨实例、跨进程 read-validate-append 原子性
- 后端无关的 `SessionTreeStorage` 协议
- 多进程竞争和 Crash Recovery 测试
- 锁超时、取消和异常退出恢复
- 长 Journal 基准；Sidecar Index 延后到真实规模需要时

### 阶段五：Harness 行为控制面——进行中

- Steering Queue、Safe Point、回执和 Session 审计——已完成基础版
- Follow-up Queue、Run Handle、失败策略和终态屏障——已完成基础版
- Continue / Resume-safe-point、Run Intent 和 Session 回放——已完成基础版
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

Steering、Follow-up 和 Continue 基础版已经完成。下一步建议实现 **Behavior Hooks**，补足
Tool Middleware 之外的 Turn / Run 行为控制，同时继续保持 Event Sink 只读。

### 8.1 具体实现范围

1. 定义独立 `BehaviorHook` 协议，不复用 `AgentEventSink` 或 `ToolMiddleware`。
2. 第一批只加入确定性的 `after_turn` 决策点，允许继续或以结构化原因停止 Run。
3. Hook 接收只读 Turn Snapshot 和 CancellationToken，不直接暴露可变 `AgentState`。
4. 多个 Hook 按注册顺序 await；异常转换为明确 Runtime Failure，不静默跳过。
5. Hook 决策必须发生在完整 Tool Result 提交之后、下一个 Provider Request 之前。
6. 后续再按真实需求增加 `before_model` 临时上下文注入，不与 `AgentContextBuilder` 职责重叠。

### 8.2 验收标准

- Event Sink 仍严格只读，现有观察者不会改变 Agent 行为。
- `after_turn` 永远在 `TurnCompleted` 对应工作完成后决定是否发起下一 Turn。
- Hook Stop 产生独立 Stop Reason、正常 `AgentFinished` 和 Session `run_finished`。
- Hook 顺序、异常、取消和慢 Hook 的 Backpressure 都有测试。
- Steering、Follow-up、Continue 和 Tool Middleware 的现有语义不发生变化。

Behavior Hooks 完成后，再进入 Parallel Tool Calls、Side-effect Policy 和 Canonical Tool
Definition。
