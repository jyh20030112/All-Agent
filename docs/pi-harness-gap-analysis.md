# SimAgentPlg 与 Pi Agent Harness 能力对照

> 更新日期：2026-07-21
>
> SimAgentPlg 基线：`9c61e77` 后的 Core 能力工作树，项目版本 `0.5.0`
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
- `after_turn` Behavior Hooks、隔离 Turn Snapshot 与结构化策略停止
- Parallel Read-only Tool Calls、显式 Side-effect 属性与确定性结果提交
- OpenAI-first Canonical `ToolDefinition`、强类型校验与旧字典兼容视图
- Python 3.12 / 3.13 CI、构建 smoke test 与 PyPI Trusted Publishing CD

因此，`BaseAgent` 已经是一个可独立使用的轻量 Harness 组装根，而不再只是
`AgentOrchestrator` 的薄包装。

与 Pi 的主要差距已经从“缺少持久化和自动压缩”转移到以下领域：

1. **行为控制面**：Steering、Follow-up、Continue 与 `after_turn` Hook 基础版已完成；
   `before_model` 等更广义行为注入仍未实现。
2. **工具调度**：只读 Tool Call 与 Canonical Definition 已完成基础版，但工具集合仍主要在
   构造或启动时确定。
3. **Provider 广度**：当前明确采用 OpenAI-first，仅实现 OpenAI-compatible Adapter；在出现
   真实需求前不建设多 Provider Schema 抽象。
4. **长 Journal 性能**：写入与读取仍会线性重建索引，需要按实际规模决定索引和归档策略。
5. **ExecutionEnv / CodeAgent**：文件、Shell、Git、Workspace、Sandbox、Approval 和 UI/RPC
   明确留给派生层，当前仓库尚未实现该层。

Session Journal 并发边界、行为控制链、安全并行调度和 OpenAI-first Canonical Tool
Definition 已补齐。下一批建议基于稳定定义实现 Dynamic Tool Set。

## 2. 当前架构

```text
BaseAgent
  ├── Active Operation / CancellationSource
  ├── Bounded Steering Queue / Control Receipt
  ├── Follow-up FIFO / Run Handle / Run-chain Gate
  ├── Continue Safe Point / Run Intent
  ├── BehaviorHook / TurnSnapshot / BehaviorDecision
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
  │     ├── Read-only Batch Scheduler / Side-effect Barrier
  │     ├── ToolDefinition / ToolEffect / OpenAI Compatibility View
  │     ├── BaseHandler / MethodToolHandler
  │     ├── McpToolHandler
  │     └── ToolMiddleware
  └── SkillManager
```

### 2.1 组件职责

| 组件 | 当前职责 |
|---|---|
| `BaseAgent` | 依赖组装、Run Chain、Steering、Follow-up、Behavior Hook、取消、空闲等待、资源生命周期、Session Restore |
| `AgentOrchestrator` | 模型—工具循环、after-Turn 决策、自动压缩、Overflow Recovery、结构化终止 |
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
- OpenAI-first `ToolDefinition`：Name、Description、Parameters、`strict` 与 Effect
- 旧 OpenAI 字典在 Handler 边界一次归一化
- 显式 `ToolEffect.READ_ONLY` / `SIDE_EFFECTING`
- Opt-in Parallel Read-only Batch 与稳定源顺序提交

默认仍按顺序执行。启用 `RuntimePolicy.parallel_tool_calls` 后，只有连续且显式标注为
`READ_ONLY` 的调用才并行；未标注、未知和 `SIDE_EFFECTING` 工具形成串行屏障。开始事件按
源顺序发出，Progress 可以交错，完成事件、Agent History 和 Session Tool Message 按 Assistant
Tool Call 源顺序提交。并行上限由 `max_parallel_tool_calls` 控制。

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

`CancellationToken` 已传递到 Model、Compactor、Tool Runtime、Middleware、Handler、Behavior Hook 和
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

### 3.7 Behavior Hooks

`BehaviorHook.after_turn()` 是独立于只读 Event Sink 和 Tool Middleware 的行为决策点。它只在
本来还会进入下一 Turn 的完整非终态回合执行，位置固定为：Tool Result 提交、
`TurnCompleted` 完成之后，下一次 `TurnStarted` 和 Provider Request 之前。

Hook 接收隔离的 `TurnSnapshot` 和当前 `CancellationToken`，不接触可变 `AgentState`。多个
Hook 按声明顺序 await；`None` 或 `BehaviorDecision.CONTINUE` 继续，首个 STOP 短路后续 Hook，
并产生 `RunStatus.COMPLETED + StopReason.BEHAVIOR_STOP`。正常 `AgentFinished` 和 Session
`run_finished` 仍会完成，因此该终态也是允许显式 Continue 的安全边界。

Hook 异常或非法返回值转换为 `RUNTIME_ERROR`，慢 Hook 形成有意的有界背压，`abort()` 可以
中断它，`wait_for_idle()` 会等待 Hook 与终态 Sink。第一批不加入 `before_model` Context 注入，
避免与 `AgentContextBuilder` 职责重叠。

### 3.8 Usage、Context 与 Compaction

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

### 3.9 Durable JSONL Session Tree

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

### 3.10 交付与兼容性

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
| Stop-after-turn Hook | `shouldStopAfterTurn` | 顺序 await `BehaviorHook.after_turn` | 基础已具备 |
| Steering | Queue + safe turn boundary | 有界 FIFO + Model-call Safe Point | 基础已具备 |
| Follow-up | Agent outer loop queue | 独立 FIFO Run Chain + Waitable Handle | 基础已具备 |
| Continue | `continue()` | 独立 Run + 无新增 User Message + Safe Point | 基础已具备 |
| Parallel Tool Calls | 默认并行，可强制顺序 | Opt-in 只读并行 + 副作用屏障 | 基础已具备，更保守 |
| Canonical Tool Definition | 内部强类型定义 | OpenAI-first `ToolDefinition` + 字典兼容 | 基础已具备 |
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

这个边界避免事件观察者意外改变运行。需要在完整 Turn 后停止时使用独立的
`BehaviorHook.after_turn`；后续扩展决策点也不应把可变 Hook 塞进 `AgentEventSink`。

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
写入 transcript。SimAgentPlg 采用更保守的双重 opt-in：Handler 必须声明 `READ_ONLY`，Agent
还必须开启并行策略。连续只读调用并发运行，但 `ToolCompleted` 与 Tool Message 在批次完成后
按源顺序提交；副作用工具不会与相邻只读批次重叠。

并行批次共享 Run Cancellation，每个 Progress 保留自己的 `tool_call_id`。普通异常转换为对应
Tool Result；外部取消会补齐批次和后续未启动调用；并行批次中的 active peer 全部收尾后，按
源顺序选择首个 COMPLETE / REJECT / CANCEL。`READ_ONLY` 是 Handler 与 Middleware 对并发安全
作出的契约，Core 不尝试从名称或参数自动推断副作用。

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

### 6.4 Canonical Tool Definition——已完成 OpenAI-first 基础版

`ToolDefinition` 已成为 Core 内部事实来源，统一保存 Name、Description、Parameters、可选
`strict` 和 `ToolEffect`。`MethodToolHandler` 同时接受强类型定义和旧 OpenAI 字典；旧字典只在
构造边界归一化一次，`.tools` 继续提供兼容的 OpenAI 请求视图。Tool Runtime 的路由、重复名称
校验、副作用调度和 Agent 日志不再解析 `tool["function"]["name"]`。

该设计明确是 OpenAI-first，不尝试抽取所有 Provider 的最低公分母。Parameters 只做 JSON
可序列化校验，具体 JSON Schema 能力仍由 OpenAI-compatible Provider 决定。未来如有真实多
Provider 需求，可以在稳定定义外增加 Adapter 转换，而无需现在预建复杂抽象。

### 6.5 Provider 广度——按需求延后

目前只有 OpenAI-compatible Adapter。这是当前产品边界，不再视为近期优先缺口；只有真实接入
需求出现后，才验证 Thinking、Cache Usage、Tool Schema 和错误分类差异。

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
- 独立于只读 Event 的 `after_turn` Behavior Hook——已完成基础版
- Queue 与 Session Journal 的进程内 / Durable 边界——已明确
- `before_model` 等扩展决策点——按真实需求延后

### 阶段六：工具调度与 Provider 扩展

- Parallel Tool Calls 与 Side-effect Policy——已完成基础版
- Canonical Tool Definition——已完成 OpenAI-first 基础版
- Dynamic Tool Set
- 第二个真实 Provider Adapter——按真实需求延后

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

Canonical Tool Definition 第一批已经完成。下一步建议实现 **Dynamic Tool Set**，让长期运行
的 Agent 可以在安全边界更新工具，而不必销毁整个 Agent 或直接修改 Handler 内部状态。

### 8.1 具体实现范围

1. 提供显式的注册、移除和替换接口，操作单位使用 `ToolDefinition` 与对应 Handler Route。
2. 第一版只允许 Agent 空闲时更新；Active Run 使用启动时快照，不接受中途改变 Tool Set。
3. Definition、Route、Middleware Chain 和 Provider Context 必须原子切换，失败时回滚旧集合。
4. 新增 Handler 先完成 Startup 再发布；移除 Handler 在旧快照不再使用后执行 Shutdown。
5. 重复名称、未知移除、替换冲突和生命周期失败必须返回明确错误，不留下部分注册状态。
6. MCP 自动刷新、基于权限的临时工具和跨进程 Tool Registry 暂不纳入第一版。

### 8.2 验收标准

- 默认静态 Handler 配置的行为完全不变。
- 更新成功后，`agent.tool_definitions`、`agent.tools`、Runtime Route 和下一次模型请求一致。
- Active Run 不观察到半更新或中途变化的工具集合。
- Startup / Shutdown / 重复名称失败会原子回滚，并有确定性生命周期测试。
- Session、Steering、Follow-up、Continue、并行调度和 Behavior Hook 语义不受影响。

完成空闲边界 Dynamic Tool Set 后，再根据真实需求决定是否支持 Run 内临时工具快照；第二个
Provider 不进入近期路线。
