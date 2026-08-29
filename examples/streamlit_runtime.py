from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import TypeVar

from ejagent.contracts import (
    AssistantMessage,
    CancellationToken,
    ControlReceipt,
    ConversationMessage,
    JsonObject,
    ModelPort,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelUsage,
    RunAudit,
    RunLimits,
    RunOutcome,
    RunResult,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolResultMessage,
    TransientInstruction,
    UserMessage,
    thaw_json_value,
)
from ejagent.harness import AgentHarness, HarnessStatus
from ejagent.storage import JsonlSessionStore
from ejagent.tools import FunctionTool, FunctionToolExecutor

_ResultT = TypeVar("_ResultT")
ModelFactory = Callable[[], ModelPort]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Configuration captured when a validation runtime starts."""

    agent_id: str = "streamlit-validation"
    store_root: Path = Path(".ejagent-sessions")
    max_turns: int = 20
    max_tokens: int | None = None
    max_repeated_tool_calls: int = 3
    probe_delay_seconds: float = 1.5

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")
        object.__setattr__(self, "agent_id", self.agent_id.strip())
        object.__setattr__(self, "store_root", Path(self.store_root).expanduser())
        if self.probe_delay_seconds <= 0:
            raise ValueError("probe_delay_seconds must be greater than zero")
        RunLimits(
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            max_repeated_tool_calls=self.max_repeated_tool_calls,
        )

    @property
    def limits(self) -> RunLimits:
        return RunLimits(
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            max_repeated_tool_calls=self.max_repeated_tool_calls,
        )


@dataclass(frozen=True, slots=True)
class ProbeExecution:
    """Timing captured for one validation tool invocation."""

    call_id: str
    tool_name: str
    started_at: datetime
    finished_at: datetime | None = None
    elapsed_seconds: float | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Detached view read by the Streamlit script thread."""

    agent_id: str
    status: HarnessStatus
    revision: int
    messages: tuple[ConversationMessage, ...]
    last_result: RunResult | None
    latest_outcome: RunOutcome | None
    audits: tuple[RunAudit, ...]
    pending_follow_ups: int
    controls: tuple[ControlReceipt, ...]
    probes: tuple[ProbeExecution, ...]
    last_error: str | None


@dataclass(slots=True)
class _ActiveProbe:
    call_id: str
    tool_name: str
    started_at: datetime
    started_monotonic: float


@dataclass(slots=True)
class _ProbeRecorder:
    active: dict[str, _ActiveProbe] = field(default_factory=dict)
    completed: list[ProbeExecution] = field(default_factory=list)

    def start(self, call: ToolCall) -> None:
        self.active[call.id] = _ActiveProbe(
            call_id=call.id,
            tool_name=call.name,
            started_at=datetime.now(UTC),
            started_monotonic=monotonic(),
        )

    def finish(self, call: ToolCall, *, cancelled: bool) -> ProbeExecution:
        active = self.active.pop(call.id)
        finished_at = datetime.now(UTC)
        execution = ProbeExecution(
            call_id=call.id,
            tool_name=call.name,
            started_at=active.started_at,
            finished_at=finished_at,
            elapsed_seconds=monotonic() - active.started_monotonic,
            cancelled=cancelled,
        )
        self.completed.append(execution)
        return execution

    def snapshot(self) -> tuple[ProbeExecution, ...]:
        running = tuple(
            ProbeExecution(
                call_id=item.call_id,
                tool_name=item.tool_name,
                started_at=item.started_at,
            )
            for item in self.active.values()
        )
        return (*self.completed, *running)


class DemoValidationModel(ModelPort):
    """Deterministic model that requests both validation tools every Run."""

    def __init__(self) -> None:
        self._batch = 0

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        cancellation.raise_if_cancelled()
        current_messages = _current_task_messages(request)
        results = tuple(
            message
            for message in current_messages
            if isinstance(message, ToolResultMessage)
        )
        usage = ModelUsage(input_tokens=12, output_tokens=8, total_tokens=20)
        if not results:
            self._batch += 1
            batch = self._batch
            yield ModelResponseCompleted(
                AssistantMessage(
                    tool_calls=(
                        ToolCall(f"demo-{batch}-a", "parallel_probe_a"),
                        ToolCall(f"demo-{batch}-b", "parallel_probe_b"),
                    )
                ),
                usage,
            )
            return

        steering = tuple(
            message.content
            for message in current_messages
            if isinstance(message, TransientInstruction)
        )
        labels = ", ".join(message.tool_name for message in results)
        text = f"Parallel validation completed with {labels}."
        if steering:
            text += f" Applied steering: {' | '.join(steering)}"
        yield ModelTextDelta(text)
        yield ModelResponseCompleted(AssistantMessage(content=text), usage)


def _current_task_messages(request: ModelRequest) -> Sequence[object]:
    for index in range(len(request.messages) - 1, -1, -1):
        if isinstance(request.messages[index], UserMessage):
            return request.messages[index + 1 :]
    return request.messages


def _validation_tools(
    recorder: _ProbeRecorder,
    delay_seconds: float,
) -> FunctionToolExecutor:
    async def probe(
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        recorder.start(call)
        try:
            await asyncio.sleep(delay_seconds)
        except asyncio.CancelledError:
            recorder.finish(call, cancelled=True)
            raise
        execution = recorder.finish(call, cancelled=False)
        return ToolExecutionResult(
            {
                "tool": call.name,
                "started_at": execution.started_at.isoformat(),
                "finished_at": execution.finished_at.isoformat()
                if execution.finished_at is not None
                else None,
                "elapsed_seconds": execution.elapsed_seconds,
            }
        )

    schema: JsonObject = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return FunctionToolExecutor(
        (
            FunctionTool(
                ToolDefinition(
                    name="parallel_probe_a",
                    description="Wait briefly and return timing data for probe A.",
                    input_schema=schema,
                ),
                probe,
            ),
            FunctionTool(
                ToolDefinition(
                    name="parallel_probe_b",
                    description="Wait briefly and return timing data for probe B.",
                    input_schema=schema,
                ),
                probe,
            ),
        )
    )


class StreamlitRuntimeController:
    """Keep one AgentHarness on a dedicated, long-lived asyncio loop."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        model_factory: ModelFactory = DemoValidationModel,
        command_timeout: float = 5.0,
    ) -> None:
        if command_timeout <= 0:
            raise ValueError("command_timeout must be greater than zero")
        self.config = config
        self._model_factory = model_factory
        self._command_timeout = command_timeout
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._harness: AgentHarness | None = None
        self._store: JsonlSessionStore | None = None
        self._recorder: _ProbeRecorder | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._follow_up_tasks: set[asyncio.Task[None]] = set()
        self._latest_outcome: RunOutcome | None = None
        self._controls: list[ControlReceipt] = []
        self._last_error: str | None = None
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"ejagent-{config.agent_id}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=command_timeout):
            raise TimeoutError("EJAgent runtime did not start in time")
        if self._startup_error is not None:
            raise RuntimeError(
                "EJAgent runtime failed to start"
            ) from self._startup_error

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def start_run(self, task: str) -> None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must not be empty")
        self._call(self._start_run(task.strip()))

    def cancel(self, reason: str = "Cancelled from Streamlit") -> bool:
        return self._call(self._cancel(reason))

    def steer(self, content: str) -> ControlReceipt:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("steering content must not be empty")
        return self._call(self._steer(content.strip()))

    def follow_up(self, task: str) -> ControlReceipt:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("follow-up task must not be empty")
        return self._call(self._follow_up(task.strip()))

    def snapshot(self) -> RuntimeSnapshot:
        return self._call(self._snapshot())

    def close(self) -> None:
        if self._closed.is_set():
            return
        loop = self._loop
        try:
            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
                future.result(timeout=max(self._command_timeout, 10.0))
        finally:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=max(self._command_timeout, 10.0))
            self._closed.set()

    def __enter__(self) -> StreamlitRuntimeController:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._initialize())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        else:
            self._ready.set()
            loop.run_forever()
        finally:
            if self._harness is not None:
                try:
                    loop.run_until_complete(self._harness.shutdown())
                except BaseException as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._closed.set()

    async def _initialize(self) -> None:
        recorder = _ProbeRecorder()
        store = JsonlSessionStore(self.config.store_root)
        harness = AgentHarness(
            agent_id=self.config.agent_id,
            model=self._model_factory(),
            tools=_validation_tools(recorder, self.config.probe_delay_seconds),
            initial_messages=(
                SystemMessage(
                    "You are validating EJAgent. When asked to verify parallel tools, "
                    "call parallel_probe_a and parallel_probe_b in one response."
                ),
            ),
            store=store,
            limits=self.config.limits,
            configuration_revision="streamlit-example-v1",
        )
        await harness.start()
        self._recorder = recorder
        self._store = store
        self._harness = harness

    async def _start_run(self, task: str) -> None:
        harness = self._require_harness()
        if harness.status is HarnessStatus.RUNNING:
            raise RuntimeError("a Run is already active; use follow-up instead")
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("a Run submission is already pending")
        self._last_error = None
        self._run_task = asyncio.create_task(self._capture_run(harness.run(task)))

    async def _capture_run(self, run: Coroutine[object, object, RunOutcome]) -> None:
        try:
            self._latest_outcome = await run
        except BaseException as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"

    async def _cancel(self, reason: str) -> bool:
        return self._require_harness().cancel(reason)

    async def _steer(self, content: str) -> ControlReceipt:
        receipt = self._require_harness().steer(content)
        self._record_control(receipt)
        return receipt

    async def _follow_up(self, task: str) -> ControlReceipt:
        handle = self._require_harness().follow_up(task)
        self._record_control(handle.receipt)
        if handle.accepted:
            watcher = asyncio.create_task(self._capture_follow_up(handle.wait()))
            self._follow_up_tasks.add(watcher)
            watcher.add_done_callback(self._follow_up_tasks.discard)
        return handle.receipt

    async def _capture_follow_up(
        self,
        outcome: Coroutine[object, object, RunOutcome],
    ) -> None:
        try:
            self._latest_outcome = await outcome
        except BaseException as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
        else:
            self._last_error = None

    async def _snapshot(self) -> RuntimeSnapshot:
        harness = self._require_harness()
        store = self._require_store()
        recorder = self._require_recorder()
        return RuntimeSnapshot(
            agent_id=harness.agent_id,
            status=harness.status,
            revision=harness.revision,
            messages=harness.messages,
            last_result=harness.last_result,
            latest_outcome=self._latest_outcome,
            audits=await store.load_audit(harness.agent_id),
            pending_follow_ups=harness.pending_follow_up_count,
            controls=tuple(self._controls),
            probes=recorder.snapshot(),
            last_error=self._last_error,
        )

    async def _shutdown(self) -> None:
        harness = self._require_harness()
        await harness.shutdown()
        tasks = tuple(self._follow_up_tasks)
        if self._run_task is not None:
            tasks = (*tasks, self._run_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _call(self, coroutine: Coroutine[object, object, _ResultT]) -> _ResultT:
        if self._closed.is_set():
            coroutine.close()
            raise RuntimeError("EJAgent runtime is closed")
        loop = self._loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise RuntimeError("EJAgent runtime is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=self._command_timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError("EJAgent runtime command timed out") from exc

    def _record_control(self, receipt: ControlReceipt) -> None:
        self._controls.append(receipt)
        del self._controls[:-20]

    def _require_harness(self) -> AgentHarness:
        if self._harness is None:
            raise RuntimeError("AgentHarness is not initialized")
        return self._harness

    def _require_store(self) -> JsonlSessionStore:
        if self._store is None:
            raise RuntimeError("JsonlSessionStore is not initialized")
        return self._store

    def _require_recorder(self) -> _ProbeRecorder:
        if self._recorder is None:
            raise RuntimeError("probe recorder is not initialized")
        return self._recorder


def message_to_data(message: ConversationMessage) -> dict[str, object]:
    """Convert a committed message into values Streamlit can render."""

    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": thaw_json_value(call.arguments),
                }
                for call in message.tool_calls
            ],
        }
    assert isinstance(message, ToolResultMessage)
    return {
        "role": "tool",
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "result": thaw_json_value(message.result),
        "is_error": message.is_error,
    }
