# SimAgentPlg

[English](README.md) | [简体中文](README_zh-CN.md)

SimAgentPlg 是一个用于构建有状态、可扩展 Agent 的轻量级 Core，面向
OpenAI-compatible 模型 API。Core 提供状态、编排、上下文、工具调度、
Middleware、MCP 和 Skill 等运行机制；Shell、文件编辑、Git、审批界面和完成工具
由具体派生 Agent 自行实现。

需要 Python 3.12 或更高版本。

## 核心能力

- 有状态的 `BaseAgent`，支持持久对话历史和 `reset()`
- Provider 无关的 `ModelAdapter` 边界，以及 OpenAI-compatible 适配器
- 公开的 `AgentOrchestrator`，负责模型—工具运行循环
- 结构化的 `AgentRunResult`、`RunStatus` 和 `StopReason`
- 显式的 `RuntimePolicy`，控制循环和完成策略
- `AgentContextBuilder`，构造不修改历史的每轮上下文
- OpenAI-first 的强类型 `ToolDefinition`，并兼容旧工具字典
- 可组合的 `BaseHandler` 和 `MethodToolHandler` 工具协议
- `ToolRuntime` 生命周期、路由、Middleware 和重复调用保护
- 通用 `ToolMiddleware` 拦截机制
- 可选的 fail-closed 工具执行策略 Middleware、审批协议和 Run 级原子调用限额
- 类型化生命周期事件、Text / Thinking Streaming 和 Tool Progress
- Run Cancellation、`abort()`、Steering Safe Point 与 `wait_for_idle()` 终态屏障
- Follow-up Run Chain、Continue Safe Point 和 `after_turn` Behavior Hooks
- 显式 Side-effect 属性与可选的只读 Tool Call 并行调度
- Provider-neutral Token Usage 与单次 Run 预算保护
- 上下文压力估算、独立窗口预算和非变异压缩准备
- 通过可插拔 `Compactor`、标准 `SummaryEntry` 和 Session 快照提供显式可取消压缩
- 可选的阈值自动压缩，以及 Provider 上下文溢出后的单次安全恢复
- 版本化 Session 序列化、追加式 JSONL Journal 和显式跨进程恢复
- 通过 `McpToolHandler` 提供可选 MCP 集成
- 通过 `SkillManager` 发现本地 Skill、投影 metadata 并显式激活上下文

Core 刻意不再内置 Bash、Git、文件系统、审批 UI 或 Finish 工具。这些能力属于
CodeAgent 等派生 Agent。

本次 Core 边界调整移除了原有的 `BashHandler`、`GitDiffHandler`、
`FinishHandler`、`HumanApproval` 和 `BashApprovalMiddleware` 公共导出；需要
这些能力的派生 Agent 应自行提供对应实现。

## 安装

```bash
uv sync
```

MCP 支持是可选能力。只有使用 MCP 的 Agent 才需要安装额外依赖：

```bash
uv sync --extra mcp
# 或：pip install "SimAgentPlg[mcp]"
```

## 配置

复制 `.env.example` 为 `.env` 并填写模型配置：

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-flash
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.7
LLM_INCLUDE_USAGE=true
```

`ModelConfig` 属于 `OpenAIModelAdapter`，不再属于 `BaseAgent`。也可以直接构造：

```python
from simagentplg import ModelConfig

config = ModelConfig(
    model="deepseek-v4-flash",
    api_key="sk-xxxxxxxx",
    base_url="https://api.deepseek.com",
)
```

接入其他模型 Provider 时，只需实现 `ModelAdapter.complete()`。适配器负责 Provider
Client 的创建、响应归一化以及可选的启动/关闭资源；`BaseAgent` 只消费归一化后的
`AssistantMessage` 协议。

## 普通 Agent

多次调用之间会保留对话历史：

```python
from simagentplg import BaseAgent, ModelConfig, OpenAIModelAdapter

agent = BaseAgent(
    OpenAIModelAdapter(ModelConfig.from_env()),
    agent_id="tutor",
    system_prompt="你是一名回答简洁的 Python 导师。",
)

first = await agent.runtime(task="请记住我更喜欢 Python。")
second = await agent.runtime(task="我更喜欢哪种编程语言？")

agent.reset()
await agent.shutdown()
```

同一个 Agent 的调用会串行执行，以保护对话状态。

## 结构化运行结果

`run()` 暴露 Core 的运行结果协议：

```python
result = await agent.run(task="解释这个仓库的架构。")

print(result.status)
print(result.stop_reason)
print(result.turns)
print(result.output)
print(result.usage.total_tokens)
print(result.usage.complete)
```

`runtime()` 继续作为兼容接口。任务完成时返回 `result.output`；运行失败、被拒绝或
取消时抛出 `AgentRunError`。

`ModelResponseCompleted` 可以携带标准化 `ModelUsage`。Usage 会保存在 Agent 内部消息
和 Session 中，但 `AgentContextBuilder` 会在构造 `llm_messages` 时移除，不会发送给
Provider。`AgentRunResult.usage` 聚合一次 Run 的所有模型请求；`complete=False` 表示
至少一次请求没有报告 Usage，不能把它当成零消耗。

### 取消正在执行的 Run

`abort()` 会取消当前模型请求、Compactor、Tool、Middleware、Behavior Hook 或 MCP 调用，
而不等待 Agent 的串行操作锁：

```python
run = asyncio.create_task(agent.run(task="执行一个较长任务"))
agent.abort("用户停止")
result = await run
```

取消返回结构化 `AgentRunResult`。`wait_for_idle()` 会一直等待到 `AgentFinished` 和全部同步
Event Sink 完成，之后同一个 Agent 可以安全复用。

### Steering 当前 Run

`steer()` 可以为正在执行的 Run 提交有界 FIFO 控制输入：

```python
receipt = agent.steer("优先检查配置文件，不要修改代码")
print(receipt.status)
```

Steering 不会中断已经开始的模型响应或 Tool Call。Core 会在下一次 Provider Context 最终
确定前的安全点按顺序应用输入，并发布 `SteeringApplied`；Run 提前结束时，未消费输入会
发布 `SteeringDiscarded`。回执明确区分已接受、Agent 空闲、队列已满和 Run 正在收尾。

### Follow-up Run Chain

`follow_up()` 把独立任务加入当前 Run Chain，并返回可等待的 Handle：

```python
handle = agent.follow_up("根据刚才的结果生成测试计划")
result = await handle.wait()
```

每个 Follow-up 都有独立的 `run_id`、事件序列和 Session Run，并等待前一个 Run 的
`AgentFinished` 及 Event Sink 完成。Queue 是进程内有界 FIFO，不伪装成持久任务系统；
失败后的丢弃或继续由 `FollowUpFailurePolicy` 显式控制。

### Continue 已有历史

`continue_run()` 从已提交历史启动新的 Run，但不会追加新的 User Message：

```python
if agent.can_continue:
    result = await agent.continue_run()
else:
    print(agent.continue_rejection_reason)
```

Continue 分配新的 `run_id`、发布 `AgentContinued`，并通过 `SessionRunIntent.CONTINUE`
持久化。只有显式安全的终态和完整 Tool Result 历史可以 Continue；其他情况会抛出带类型化
原因的 `ContinueRejectedError`。

### Behavior Hooks

`BehaviorHook.after_turn()` 决定一个非终态完整 Turn 是否允许进入下一次 Provider 请求：

```python
from simagentplg import BehaviorDecision


class StopAfterFirstToolTurn:
    async def after_turn(self, snapshot, *, cancellation):
        if snapshot.turn >= 1:
            return BehaviorDecision.stop("在完整 Turn 边界暂停")
        return None


agent = BaseAgent(
    model,
    agent_id="controlled-agent",
    behavior_hooks=[StopAfterFirstToolTurn()],
)
```

决策点位于所有 Tool Result 和 `TurnCompleted` 提交之后、下一次 `TurnStarted` 之前。Hook
接收隔离的 `TurnSnapshot`，不能修改 `AgentState`；多个 Hook 按声明顺序执行，首个 STOP
短路后续 Hook，并以 `StopReason.BEHAVIOR_STOP` 完成 Run。该终态可以显式 Continue。

### 并行只读工具

并行 Tool Call 需要工具和 Agent 两侧同时显式开启：Handler 必须把工具声明为
`ToolEffect.READ_ONLY`，Runtime Policy 也必须允许并行：

```python
from simagentplg import (
    MethodToolHandler,
    RuntimePolicy,
    ToolDefinition,
    ToolEffect,
)

LOOKUP_TOOL = ToolDefinition(
    name="lookup",
    description="查询一个值。",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    effect=ToolEffect.READ_ONLY,
)


class LookupHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((LOOKUP_TOOL,))


agent = BaseAgent(
    model,
    agent_id="parallel-lookups",
    handlers=[LookupHandler()],
    runtime_policy=RuntimePolicy(
        parallel_tool_calls=True,
        max_parallel_tool_calls=4,
    ),
)
```

调度器只并行连续的只读调用。未标注、未知和 `SIDE_EFFECTING` 工具保持顺序执行，并在只读
批次之间形成屏障；默认策略仍是完全串行，因此升级不会改变现有 Handler 行为。

并行批次的 `ToolStarted` 按源顺序发布，Progress 可以按 `tool_call_id` 交错。Handler 可以
并发完成，但 `ToolCompleted`、Agent History 和 Session Tool Message 始终按 Assistant
Tool Call 的原始顺序提交。外部取消会补齐活动和待执行调用；出现终态 Tool Control 时，
活动 peer 先安全收尾，再由源顺序中的首个终态结果停止 Run。`READ_ONLY` 是 Handler 及其
Middleware 作出的并发安全契约，Core 不会自动推断副作用。

### 流式模型响应

`BaseAgent.run()` 仍只返回最终 `AgentRunResult`；临时文本和推理片段通过
`AssistantTextDelta`、`AssistantThinkingDelta` 事件观察。Provider Adapter 负责组装完整
Tool Call，只有完整 `AssistantMessage` 才会进入 Agent State。Delta 不写入 Session，
只实现 `complete()` 的现有 Adapter 仍可通过默认 `stream()` 回退工作。

## 上下文压力与压缩准备

Context Window 容量和累计 Run 消耗是两个独立概念。可以为 Agent 配置可选的
`CompactionPolicy`，在每次模型请求前评估完整 Provider 上下文：

```python
from simagentplg import CompactionPolicy, ContextBudget

context_policy = CompactionPolicy(
    ContextBudget(
        context_window=128_000,
        reserve_tokens=16_000,
        keep_recent_tokens=20_000,
    )
)

agent = BaseAgent(
    model,
    agent_id="context-aware",
    compaction_policy=context_policy,
)
```

评估会组合最近一次 Assistant `ModelUsage`、它之后新增的消息，以及包含当前 Tool Schema
的 UTF-8 感知保守估算。配置策略后，每轮都会发布 `ContextPressureEvaluated`；达到阈值
时，事件中的 `CompactionPreparation` 会分离受保护消息、待摘要的旧完整
User/Assistant/Tool Turn，以及需要原文保留的最近 Turn。Tool Call 和对应 Tool Result
不会被切开。

只配置 `CompactionPolicy` 时，压力评估仍是只读观察。应用也可以直接调用
`estimate_context_usage()` 和 `prepare_compaction()`，并通过
`MessageTokenEstimator` 替换默认估算器。

## 自动压缩与 Overflow 恢复

自动行为默认关闭；启用时复用同一个 `CompactionPolicy` 和 `Compactor`：

```python
from simagentplg import AutoCompactionPolicy

agent = BaseAgent(
    model,
    agent_id="context-aware",
    compaction_policy=context_policy,
    compactor=my_compactor,
    auto_compaction_policy=AutoCompactionPolicy(),
)
```

达到压力阈值后，Core 会在同一次 Agent Run 内压缩旧完整 Turn、重建上下文，再请求模型。
Provider Adapter 抛出 `ContextOverflowError` 时，Core 最多执行一次“压缩—重建—重试”。
第二次溢出返回 `StopReason.CONTEXT_OVERFLOW`；Compactor 失败返回
`StopReason.COMPACTION_FAILED`。一旦 Text 或 Thinking Delta 已对外发布，Core 不会重试，
从而避免重复的流式输出。

`AutoCompactionPolicy(compact_on_pressure=False)` 可以只保留 Overflow 恢复；省略该策略或
设置 `enabled=False` 会关闭全部自动行为。Provider Adapter 通过 `ModelProviderError` 和
`ModelErrorKind` 统一区分上下文溢出、限流、超时、认证及普通 Provider 错误。

## 显式上下文压缩

派生 Agent 通过可取消的 `Compactor` 协议提供摘要行为，然后显式调用：

```python
agent = BaseAgent(
    model,
    agent_id="context-aware",
    compaction_policy=context_policy,
    compactor=my_compactor,
)

compaction = await agent.compact()
print(compaction.status)
print(compaction.summary)
```

`ModelCompactor` 可以把借用的 `ModelAdapter` 接入该协议，同时由应用继续拥有摘要 Prompt：

```python
compactor = ModelCompactor(
    summary_model,
    context_builder=build_summary_context,
    source="summary-model:v1",
)
```

注入的 Builder 接收 `CompactionRequest`，返回完整 `ContextBuildResult`。调用方负责借用模型
的生命周期，因此 Core 不会静默创建另一个 Provider Client，也不会替应用选择 Prompt。

Core 将 `CompactionRequest` 交给 Compactor，由 Core 在 `SummaryEntry` 中写入可信的范围和
Token metadata，最后原子替换成“受保护消息 + Summary + 最近 Turn”。失败或取消返回
结构化 `CompactionResult`，历史保持不变。重复压缩时，旧 Summary 会传给 Compactor
合并，并由新 Summary 消息替换。

生命周期通过 `CompactionStarted`、`CompactionCompleted` 和 `CompactionFailed` 发布。
`abort()`、`wait_for_idle()` 同时适用于普通 Run 和压缩。`SessionRecorder` 保存紧凑恢复
快照，同时保留原始 `SessionMessage` 审计条目。每次压缩都有独立 `operation_id` 和
`CompactionTrigger`。Core 不会替派生 Agent 选择摘要模型或 Prompt。

## 持久化 Session Journal

`SessionRecorder` 可以使用 `JsonlSessionStorage`，为每个已接受的生命周期变更追加一条
版本化语义 Record：

```python
from simagentplg import JsonlSessionStorage, SessionRecorder

storage = JsonlSessionStorage("./sessions")
recorder = SessionRecorder(session_id="project-42", storage=storage)
agent = BaseAgent(model, agent_id="core-agent", event_sink=recorder)
await agent.run(task="remember this decision")
```

另一个进程可以读取完成快照并显式恢复新的 Agent：

```python
saved = await storage.load("project-42")
if saved is not None:
    resumed = BaseAgent(model, agent_id="core-agent", event_sink=recorder)
    resumed.restore_session(saved)
```

每条 JSONL Record 都包含单调递增的 `revision`、不可变 `record_id`、`parent_id` 和
`branch_id`。文件顺序定义全局 Revision，父指针定义逻辑树。`SessionRecorder` 追加 `run_started`、`message_appended`、
`compaction_applied`、`run_finished` 等紧凑 Mutation；显式 `save()` 用完整 Checkpoint
支持导入和导出。

Branch 会复用源历史，不复制或改写旧 Record：

```python
forked = await storage.fork("project-42", branch_id="experiment")
rolled_back = await storage.rollback(
    "project-42",
    to_record_id="a-completed-ancestor-record",
    branch_id="rollback-before-change",
)
retry = await storage.prepare_retry(
    "project-42",
    run_id="run-to-repeat",
    branch_id="retry-run",
)
```

`fork()` 在已完成的投影上创建通用 Branch；`rollback()` 要求目标是源 Head 的祖先；
`prepare_retry()` 在指定 Run 之前创建 Branch 并返回原始 Task。Core 不会自动执行重试，
因为 Tool Call 可能已经产生外部副作用。可以使用 `checkout()`、`head()` 和
`list_branches()` 检查 Session 树。继续某个 Branch 时，需要恢复其 Checkout，并把相同
的 `branch_id` 传给 `SessionRecorder`。

Session ID 会映射为哈希文件名。每一行先完整编码，再通过一次追加写入并执行 `fsync`；
中断产生的不完整尾行会在读取时忽略，并在下一次追加前修复。已经换行的损坏 JSON 或
未知 Journal Schema 会抛出 `SessionSerializationError`，不会被误认为 Session 不存在。

`JsonlSessionStorage` 使用稳定的 `.jsonl.lock` Sidecar，把进程内协调和 POSIX Advisory
Lock 组合在同一个 read-validate-append 事务中。不同 Storage 实例和不同进程不会分配重复
Revision 或静默覆盖 Branch Head；过期 Head 返回 `SessionConflictError`，锁超时返回
`SessionLockTimeoutError`。锁等待在线程中执行，不阻塞 Event Loop。

`restore_session()` 会校验 Agent 身份，并拒绝包含未完成 Run 的 Session。Core 不会重放
中断的 Tool Call，因为它可能已经产生外部副作用。文件锁契约面向本地 POSIX 文件系统；
网络文件系统需要验证其 Advisory Lock 语义，或提供其他 `SessionTreeStorage` Backend。

## RuntimePolicy

工具是否存在和任务是否必须显式完成已经解耦：

```python
from simagentplg import RuntimePolicy

policy = RuntimePolicy(
    max_steps=20,
    max_no_tool_responses=3,
    max_repeated_tool_calls=3,
    max_run_tokens=None,
    require_explicit_finish=False,
    parallel_tool_calls=False,
    max_parallel_tool_calls=None,
)
```

`parallel_tool_calls` 默认关闭；开启后也只并行显式声明为 `READ_ONLY` 的工具。
`max_parallel_tool_calls` 可以限制每个只读批次的并发量。

可选的 `max_run_tokens` 在轮次边界阻止下一次模型请求。当前响应及其请求的工具会先完整
收尾；达到预算时返回 `TOKEN_BUDGET_EXCEEDED`，需要继续但 Provider 未报告 Usage 时
返回 `USAGE_UNAVAILABLE`。

默认情况下，Agent 可以调用工具，之后用普通文本完成任务。自主型派生 Agent 可以要求
必须调用完成工具：

```python
policy = RuntimePolicy(require_explicit_finish=True)
```

此时派生 Agent 必须自行注册一个返回 `ToolControl.COMPLETE` 的工具。

## 自定义工具

工具通过 Handler 组织。`MethodToolHandler` 会把名为 `add` 的工具映射到异步
`do_add()` 方法：

```python
from collections.abc import Mapping
from typing import Any

from simagentplg import (
    MethodToolHandler,
    StepOutcome,
    ToolDefinition,
    ToolEffect,
)

ADD_TOOL = ToolDefinition(
    name="add",
    description="计算两个数的和。",
    parameters={
        "type": "object",
        "properties": {
            "left": {"type": "number"},
            "right": {"type": "number"},
        },
        "required": ["left", "right"],
        "additionalProperties": False,
    },
    effect=ToolEffect.READ_ONLY,
    strict=True,
)


class MathHandler(MethodToolHandler):
    def __init__(self) -> None:
        super().__init__((ADD_TOOL,))

    async def do_add(self, arguments: Mapping[str, Any]) -> StepOutcome:
        return StepOutcome(
            {"value": arguments["left"] + arguments["right"]}
        )
```

显式注册到 Agent：

```python
agent = BaseAgent(
    OpenAIModelAdapter(ModelConfig.from_env()),
    agent_id="calculator",
    handlers=[MathHandler()],
)
```

重复工具名会在启动阶段报错，不会静默覆盖。`ToolDefinition` 统一保存 OpenAI Function 的
名称、描述、Parameters、可选 `strict` 和执行 Effect；`to_openai_tool()` 只输出 Provider
请求字段，不会泄漏 Core 专用的 Effect metadata。

原有 OpenAI function-calling 字典仍然兼容，并在 Handler 构造时归一化一次。兼容写法不会
立即弃用，现有项目不需要同步迁移：

```python
MethodToolHandler((OPENAI_TOOL_DICTIONARY,))
```

### 工具执行进度

长时间运行的工具可以选择声明一个作用域限定的 `progress` Reporter。没有声明该参数的
现有 `do_*` 方法仍然兼容：

```python
from simagentplg import ToolProgressReporter, ToolProgressUpdate


async def do_index(
    self,
    arguments,
    *,
    cancellation,
    progress: ToolProgressReporter | None = None,
) -> StepOutcome:
    if progress is not None:
        await progress.report(
            ToolProgressUpdate(
                "正在建立文件索引",
                {"completed": 12, "total": 40},
            )
        )
    return StepOutcome({"indexed": 40})
```

每条有效更新都会生成关联当前 run、turn 和 tool call 的 `ToolProgressed` 事件。
Progress 保持顺序，在取消或 `ToolCompleted` 后停止接收；它不会改变 `StepOutcome`、
`ToolControl`，也不会写入 Agent State、Session 或模型上下文。

### 工具控制信号

工具结果数据与运行控制已经分离：

```python
from simagentplg import StepOutcome, ToolControl

StepOutcome(data)  # 继续模型—工具循环
StepOutcome(data, control=ToolControl.COMPLETE)
StepOutcome(data, control=ToolControl.REJECT)
StepOutcome(data, control=ToolControl.CANCEL)
```

运行时由此可以区分正常完成、策略拒绝和取消。

## Tool Middleware

`ToolMiddleware` 是工具执行外围唯一的拦截链：

```python
from simagentplg import ToolMiddleware


class AuditMiddleware(ToolMiddleware):
    async def __call__(self, context, call_next):
        print("before", context.tool_name)
        result = await call_next(context)
        print("after", context.tool_name)
        return result
```

可选的 `ToolPolicyMiddleware` 是建立在同一条链上的官方具体 Middleware；它不会增加第二条
Policy 通道，也不会给 `BaseAgent` 增加特殊参数：

```python
from simagentplg import (
    RuleBasedToolPolicy,
    ToolApprovalDecision,
    ToolEffect,
    ToolPolicyAction,
    ToolPolicyMiddleware,
    ToolPolicyRule,
)


class ConsoleApprover:
    async def approve(self, request):
        # 实际应用可以把 Request 转发给自己的 UI 或 RPC 层。
        return ToolApprovalDecision(approved=False, reason="操作员拒绝")


policy = RuleBasedToolPolicy(
    (
        ToolPolicyRule(
            rule_id="approve-side-effects",
            action=ToolPolicyAction.REQUIRE_APPROVAL,
            effects=frozenset({ToolEffect.SIDE_EFFECTING}),
            max_calls_per_run=5,
            reason="该工具可能修改外部状态",
        ),
        ToolPolicyRule(
            rule_id="allow-reads",
            action=ToolPolicyAction.ALLOW,
            effects=frozenset({ToolEffect.READ_ONLY}),
            max_calls_per_run=20,
        ),
    ),
    default_action=ToolPolicyAction.DENY,
)

agent = BaseAgent(
    model,
    agent_id="policy-agent",
    handlers=[handler],
    middlewares=[
        ToolPolicyMiddleware(policy, approver=ConsoleApprover()),
        AuditMiddleware(),
    ],
)
```

规则按声明顺序采用 first-match 语义，可以匹配精确工具名、`ToolEffect`，也可以提供同步或
异步的 `when(context)` 条件。`max_calls_per_run` 会原子预留调用次数，能覆盖并行只读调用，
并在每个 Agent Run 开始时重置。审批默认 fail-closed：没有 Approver、发生异常、返回非法
结果或明确拒绝时，都会返回 `ToolControl.REJECT`，Handler 绝不会执行。拒绝载荷包含工具名、
安全的原因和匹配的 `rule_id`，不会回显参数。

Core 只提供策略和审批协议，不提供审批 UI，也不内置 Shell、文件系统等领域风险规则；这些
仍由具体应用负责。

## MCP 工具

MCP 使用相同的 Handler 协议：

```python
from simagentplg import (
    BaseAgent,
    McpToolHandler,
    ModelConfig,
    OpenAIModelAdapter,
)

agent = BaseAgent(
    OpenAIModelAdapter(ModelConfig.from_env()),
    agent_id="browser",
    handlers=[McpToolHandler("examples/mcp_config.json")],
)
```

启用 MCP 的 Agent 可以执行 MCP 工具，然后直接用普通文本完成任务。只有
`RuntimePolicy` 明确要求时，才需要额外的完成工具。

## Skill

Skill 是独立于 Handler 工具的提示词和资源扩展：

```python
from pathlib import Path

from simagentplg import BaseAgent, ModelConfig, OpenAIModelAdapter

agent = BaseAgent(
    OpenAIModelAdapter(ModelConfig.from_env()),
    agent_id="skilled-agent",
    skills_dir=Path("examples/skills"),
)
```

`SkillManager` 会发现包含 `SKILL.md` 的子目录，并注入包含名称、描述和文件位置的
紧凑 metadata。用户可以用 `$skill_name` 或 `skill:skill_name` 显式选择 Skill，
将其完整指令注入当前上下文。Core 不注册特殊的 Skill 工具；未来带文件读取工具的
派生 Agent 可以根据 metadata 中的位置渐进加载 Skill。

```text
examples/skills/
  release_notes/
    SKILL.md
    template.md
    examples/
      sample.md
```

## Core 边界

SimAgentPlg Core 负责机制：

```text
Orchestration + State + Context + Runtime Policy + Run Result
+ Model Adapter + Tool Protocol + Middleware + MCP + Skills
+ Lifecycle Events + Session + Streaming + Tool Progress + Usage Budget
+ Context Pressure + Compaction Preparation
+ Model Compactor + Summary Entry + Durable Session Journal + Session Tree
+ Cancellation + Steering + Follow-up + Continue + Behavior Hooks
+ Parallel Read-only Tool Scheduler + Side-effect Barrier
+ Canonical Tool Definition + OpenAI Schema Compatibility View
```

派生 Agent 负责具体能力与策略：

```text
Shell + Filesystem + Git + Workspace + Approval UI
+ Sandbox + Completion Tool + Product Interface
```

架构分析和后续路线参见
[Pi Harness 对照分析](docs/pi-harness-gap-analysis.md)。

## 示例

```bash
uv run python examples/01_stateful_chat.py
uv run python examples/02_custom_tool.py
uv run python examples/04_mcp_tools.py
uv run python examples/06_skill.py
uv run python examples/13_usage_budget.py
uv run python examples/14_context_pressure.py
uv run python examples/15_explicit_compaction.py
uv run python examples/16_durable_session.py record
uv run python examples/16_durable_session.py resume
```

## 测试

```bash
uv run python -m unittest discover -s tests -p 'test*.py' -q
```

提交变更前运行完整的本地质量门：

```bash
uv sync --locked --all-extras --group dev
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy
uv build
```

## 发布

PyPI 发布由 `.github/workflows/release.yml` 和 Trusted Publishing 完成，GitHub
中不保存长期 API Token。配置 `pypi` Environment 和 PyPI Publisher 后，先把发布提交
合并到 `main`，再推送与项目版本一致的 Tag：

```text
PyPI 项目：SimAgentPlg
GitHub Owner：jyh20030112
Repository：SimAgentPlg
Workflow：release.yml
Environment：pypi
```

建议为 GitHub `pypi` Environment 配置 Required Reviewer，并限制只有 Maintainer 可以
创建 `v*` Tag。然后执行：

```bash
git tag v0.5.0
git push origin v0.5.0
```

工作流会拒绝不在 `main` 上或与 `project.version` 不一致的 Tag，重新执行完整质量矩阵，
构建并 smoke test 发布产物，最后使用短期 OIDC 身份上传 PyPI。

## 公共 API

包根目录导出：

- Agent：`BaseAgent`、`AgentOrchestrator`、`AgentState`、`AgentStatus`、`BehaviorHook`、`BehaviorDecision`、`BehaviorAction`、`TurnSnapshot`
- Provider：`ModelAdapter`、`OpenAIModelAdapter`、`ModelConfig`、`AssistantMessage`、`ModelToolCall`、`ModelUsage`、`ModelErrorKind`、`ModelProviderError`、`ContextOverflowError`、`ModelRateLimitError`、`ModelTimeoutError`、`ModelAuthenticationError`
- Runtime：`RuntimePolicy`、`AgentRunResult`、`RunUsage`、`AgentRunError`、`RunStatus`、`StopReason`
- 控制面：`ControlInput`、`ControlReceipt`、`ControlStatus`、`FollowUpHandle`、`FollowUpFailurePolicy`、`ContinueRejectedError`、`ContinueRejectedReason`
- Session：`AgentSession`、`SessionRecorder`、`SessionStorage`、`SessionJournalStorage`、`SessionTreeStorage`、`MemorySessionStorage`、`JsonlSessionStorage`、`SessionRunIntent`、`SessionCompaction`、`SessionRecord`、`SessionRecordDraft`、`SessionRecordKind`、`SessionBranchIntent`、`SessionBranch`、`SessionCheckout`、`SessionRetry`、`DEFAULT_SESSION_BRANCH`、`SESSION_SCHEMA_VERSION`、`SESSION_JOURNAL_SCHEMA_VERSION`、`session_to_dict`、`session_from_dict`、`SessionError`、`SessionSerializationError`、`SessionStorageError`、`SessionConflictError`、`SessionLockTimeoutError`
- Context：`AgentContextBuilder`、`ContextBuildResult`、`ContextBudget`、`ContextUsageEstimate`、`CompactionPolicy`、`AutoCompactionPolicy`、`CompactionDecision`、`CompactionPreparation`、`MessageTokenEstimator`
- Compaction：`CompactionRuntime`、`Compactor`、`ModelCompactor`、`CompactionContextBuilder`、`CompactorOutput`、`CompactionRequest`、`CompactionResult`、`CompactionStatus`、`CompactionTrigger`、`SummaryEntry`
- Tool：`ToolDefinition`、`ToolDefinitionError`、`ToolEffect`、`StepOutcome`、`ToolControl`、`ToolProgressReporter`、`ToolProgressUpdate`、`BaseHandler`、`MethodToolHandler`、`McpToolHandler`
- Middleware：`Middleware`、`ToolMiddleware`、`ToolCallContext`、`ToolNext`、`ToolExecutionPolicy`、`RuleBasedToolPolicy`、`ToolPolicyRule`、`ToolPolicyPredicate`、`ToolPolicyAction`、`ToolPolicyDecision`、`ToolPolicyMiddleware`、`ToolApprover`、`ToolApprovalRequest`、`ToolApprovalDecision`
- Event：`AgentEvent`、`AgentEventSink`、`AgentStarted`、`AgentContinued`、`TurnStarted`、`MessageCompleted`、`ToolStarted`、`ToolProgressed`、`ToolCompleted`、`TurnCompleted`、`AgentFinished`
- 扩展：`McpServerManager`、`SkillManager`

## License

MIT
