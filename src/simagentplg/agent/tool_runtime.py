import asyncio
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from simagentplg.agent.cancellation import (
    AgentCancelledError,
    CancellationSource,
    CancellationToken,
)
from simagentplg.agent.events import (
    AgentEventEmitter,
    ToolCompleted,
    ToolProgressed,
    ToolStarted,
)
from simagentplg.agent.state import AgentState
from simagentplg.agent.types import (
    StepOutcome,
    ToolCallResult,
    ToolControl,
    ToolProgressReporter,
    ToolProgressUpdate,
)
from simagentplg.handlers.base import BaseHandler, ToolEffect
from simagentplg.middleware import (
    ToolCallContext,
    ToolMiddleware,
    ToolNext,
    compose_tool_middlewares,
)
from simagentplg.providers.base import ModelToolCall


class RepeatedToolCallError(RuntimeError):
    """Raised when an identical tool call reaches the configured limit."""


class _ScopedToolProgressReporter(ToolProgressReporter):
    """Serialize updates and reject them after one tool call settles."""

    def __init__(
        self,
        *,
        event_emitter: AgentEventEmitter,
        turn: int,
        tool_call: ModelToolCall,
        cancellation: CancellationToken,
    ) -> None:
        self._event_emitter = event_emitter
        self._turn = turn
        self._tool_call = tool_call
        self._cancellation = cancellation
        self._lock = asyncio.Lock()
        self._accepting = True

    async def report(self, update: ToolProgressUpdate) -> None:
        if not isinstance(update, ToolProgressUpdate):
            raise TypeError(
                "tool progress reporter requires ToolProgressUpdate, "
                f"got {type(update).__name__}"
            )

        async with self._lock:
            if not self._accepting or self._cancellation.cancelled:
                return
            emission = asyncio.create_task(
                self._event_emitter.emit(
                    ToolProgressed(
                        self._turn,
                        self._tool_call,
                        update,
                    )
                )
            )
            try:
                await asyncio.shield(emission)
            except asyncio.CancelledError:
                await asyncio.gather(emission, return_exceptions=True)
                raise

    async def close(self) -> None:
        """Wait for an in-flight update, then reject future reports."""

        async with self._lock:
            self._accepting = False


class ToolRuntime:
    """Lifecycle, routing, middleware, and execution for tool handlers."""

    def __init__(
        self,
        handlers: Iterable[BaseHandler],
        middlewares: Iterable[ToolMiddleware],
        *,
        state: AgentState,
        logger: logging.Logger,
        event_emitter: AgentEventEmitter,
        max_repeated_tool_calls: int = 3,
    ) -> None:
        if max_repeated_tool_calls <= 0:
            raise ValueError("max_repeated_tool_calls must be greater than zero")
        self.handlers = list(handlers)
        self.middlewares = list(middlewares)
        self.state = state
        self.logger = logger
        self.event_emitter = event_emitter
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self._tool_routes: dict[str, BaseHandler] = {}
        self._active_middlewares: list[ToolMiddleware] = []
        self._tool_chain: ToolNext | None = None
        self._started = False
        self._last_tool_signature: tuple[str, str] | None = None
        self._repeated_tool_calls = 0

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [tool for handler in self.handlers for tool in handler.tools]

    async def startup(self) -> None:
        if self._started:
            return

        started_handlers: list[BaseHandler] = []
        started_middlewares: list[ToolMiddleware] = []
        try:
            for handler in self.handlers:
                await handler.startup()
                started_handlers.append(handler)
            self._tool_routes = self._build_tool_routes()
            self._active_middlewares = (
                [middleware for middleware in self.middlewares if middleware.enabled]
                if self._tool_routes
                else []
            )
            for middleware in self._active_middlewares:
                await middleware.startup()
                started_middlewares.append(middleware)
            self._tool_chain = compose_tool_middlewares(
                self._active_middlewares,
                self._dispatch_handler,
            )
        except Exception:
            for middleware in reversed(started_middlewares):
                try:
                    await middleware.shutdown()
                except Exception as shutdown_error:
                    self.logger.warning(
                        "Middleware %s rollback shutdown failed: %s",
                        type(middleware).__name__,
                        shutdown_error,
                    )
            for handler in reversed(started_handlers):
                try:
                    await handler.shutdown()
                except Exception as shutdown_error:
                    self.logger.warning(
                        "Handler %s rollback shutdown failed: %s",
                        type(handler).__name__,
                        shutdown_error,
                    )
            self._tool_routes.clear()
            self._active_middlewares.clear()
            self._tool_chain = None
            raise

        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return

        errors: list[Exception] = []
        for middleware in reversed(self._active_middlewares):
            try:
                await middleware.shutdown()
            except Exception as exc:
                errors.append(exc)
        for handler in reversed(self.handlers):
            try:
                await handler.shutdown()
            except Exception as exc:
                errors.append(exc)

        self._tool_routes.clear()
        self._active_middlewares.clear()
        self._tool_chain = None
        self._started = False
        if errors:
            raise RuntimeError(
                f"failed to shut down {len(errors)} handler(s)"
            ) from errors[0]

    async def on_task_start(self) -> None:
        """Reset per-task tool state and notify handlers/middleware."""

        self._last_tool_signature = None
        self._repeated_tool_calls = 0
        for handler in self.handlers:
            await handler.on_task_start()
        for middleware in self._active_middlewares:
            await middleware.on_task_start()

    async def dispatch(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        tool_call_id: str | None = None,
        cancellation: CancellationToken | None = None,
        progress: ToolProgressReporter | None = None,
    ) -> StepOutcome:
        if not self._started:
            await self.startup()

        self._get_handler(tool_name)
        if self._tool_chain is None:
            raise RuntimeError("tool middleware chain is not initialized")
        token = cancellation or CancellationSource().token
        context = ToolCallContext(
            state=self.state,
            tool_name=tool_name,
            arguments=dict(arguments),
            tool_call_id=tool_call_id,
            cancellation=token,
            progress=progress,
        )
        return await token.run(self._tool_chain(context))

    async def execute_tool_call(
        self,
        tool_call: ModelToolCall,
        *,
        cancellation: CancellationToken,
        prepared: bool = False,
        defer_completion: bool = False,
    ) -> ToolCallResult:
        tool_name = tool_call.name
        raw_arguments = tool_call.arguments
        if not prepared:
            await self.event_emitter.emit(ToolStarted(self.state.turn, tool_call))
        progress = _ScopedToolProgressReporter(
            event_emitter=self.event_emitter,
            turn=self.state.turn,
            tool_call=tool_call,
            cancellation=cancellation,
        )
        error: str | None = None
        immediate_result: ToolCallResult | None = None
        repeated_error: RepeatedToolCallError | None = None
        outcome: StepOutcome | None = None
        try:
            try:
                if not prepared:
                    self._check_repeated_tool_call(tool_name, raw_arguments)
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be a JSON object")
                outcome = await self.dispatch(
                    tool_name,
                    arguments,
                    tool_call_id=tool_call.id,
                    cancellation=cancellation,
                    progress=progress,
                )
            except AgentCancelledError as exc:
                immediate_result = self._cancelled_tool_result(
                    tool_call,
                    str(exc),
                )
            except RepeatedToolCallError as exc:
                immediate_result = ToolCallResult((), error=str(exc))
                repeated_error = exc
            except Exception as exc:
                error = str(exc)
                outcome = StepOutcome(
                    {
                        "status": "error",
                        "tool": tool_name,
                        "error": error,
                    }
                )
        finally:
            await progress.close()

        if immediate_result is not None:
            if not defer_completion:
                await self.complete_tool_call(tool_call, immediate_result)
            if repeated_error is not None:
                raise repeated_error
            return immediate_result

        assert outcome is not None
        serialized = serialize_tool_result(outcome.data)
        message = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": serialized,
        }
        if outcome.control is not ToolControl.CONTINUE:
            result = ToolCallResult(
                (message,),
                control=outcome.control,
                output=serialized,
                error=error,
            )
        else:
            result = ToolCallResult((message,), error=error)
        if not defer_completion:
            await self.complete_tool_call(tool_call, result)
        return result

    async def prepare_parallel_tool_calls(
        self,
        tool_calls: Sequence[ModelToolCall],
    ) -> bool:
        """Reserve repetition state and emit ordered starts for one safe batch."""

        if not self._reserve_repeated_tool_calls(tool_calls):
            return False
        for tool_call in tool_calls:
            await self.event_emitter.emit(ToolStarted(self.state.turn, tool_call))
        return True

    def can_parallelize_tool_calls(
        self,
        tool_calls: Sequence[ModelToolCall],
    ) -> bool:
        """Return whether the full source group can reserve repetition state."""

        return self._repetition_state_after(tool_calls) is not None

    async def complete_tool_call(
        self,
        tool_call: ModelToolCall,
        result: ToolCallResult,
    ) -> None:
        """Emit one completion at the scheduler's deterministic commit point."""

        await self.event_emitter.emit(ToolCompleted(self.state.turn, tool_call, result))

    def tool_effect(self, tool_name: str) -> ToolEffect:
        """Return one route's declared effect, defaulting unknown calls to unsafe."""

        handler = self._tool_routes.get(tool_name)
        if handler is None:
            return ToolEffect.SIDE_EFFECTING
        return handler.tool_effect(tool_name)

    async def cancel_tool_call(
        self,
        tool_call: ModelToolCall,
        *,
        reason: str,
    ) -> ToolCallResult:
        """Settle an unstarted call skipped after run cancellation."""

        await self.event_emitter.emit(ToolStarted(self.state.turn, tool_call))
        result = self._cancelled_tool_result(tool_call, reason)
        await self.event_emitter.emit(ToolCompleted(self.state.turn, tool_call, result))
        return result

    def _build_tool_routes(self) -> dict[str, BaseHandler]:
        routes: dict[str, BaseHandler] = {}
        for handler in self.handlers:
            for tool_name in handler.tool_names:
                if tool_name in routes:
                    first = type(routes[tool_name]).__name__
                    second = type(handler).__name__
                    raise ValueError(
                        f"duplicate tool {tool_name!r} in {first} and {second}"
                    )
                routes[tool_name] = handler
        return routes

    def _get_handler(self, tool_name: str) -> BaseHandler:
        try:
            return self._tool_routes[tool_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._tool_routes)) or "none"
            raise KeyError(
                f"unknown tool {tool_name!r}; available tools: {available}"
            ) from exc

    async def _dispatch_handler(self, context: ToolCallContext) -> StepOutcome:
        handler = self._get_handler(context.tool_name)
        return await handler.execute(context)

    def _cancelled_tool_result(
        self,
        tool_call: ModelToolCall,
        reason: str,
    ) -> ToolCallResult:
        message = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": serialize_tool_result(
                {
                    "status": "cancelled",
                    "tool": tool_call.name,
                    "error": reason,
                }
            ),
        }
        return ToolCallResult(
            (message,),
            error=reason,
            cancelled=True,
        )

    def _check_repeated_tool_call(
        self,
        tool_name: str,
        raw_arguments: str,
    ) -> None:
        signature = self._tool_signature(tool_name, raw_arguments)
        if signature == self._last_tool_signature:
            self._repeated_tool_calls += 1
        else:
            self._last_tool_signature = signature
            self._repeated_tool_calls = 1

        if self._repeated_tool_calls >= self.max_repeated_tool_calls:
            raise RepeatedToolCallError(
                f"tool {tool_name!r} was called with the same arguments "
                f"{self.max_repeated_tool_calls} consecutive times"
            )

    def _reserve_repeated_tool_calls(
        self,
        tool_calls: Sequence[ModelToolCall],
    ) -> bool:
        """Atomically reserve source-ordered guard state or request fallback."""

        next_state = self._repetition_state_after(tool_calls)
        if next_state is None:
            return False
        self._last_tool_signature, self._repeated_tool_calls = next_state
        return True

    def _repetition_state_after(
        self,
        tool_calls: Sequence[ModelToolCall],
    ) -> tuple[tuple[str, str] | None, int] | None:
        last_signature = self._last_tool_signature
        repeated_calls = self._repeated_tool_calls
        for tool_call in tool_calls:
            signature = self._tool_signature(
                tool_call.name,
                tool_call.arguments,
            )
            if signature == last_signature:
                repeated_calls += 1
            else:
                last_signature = signature
                repeated_calls = 1
            if repeated_calls >= self.max_repeated_tool_calls:
                return None
        return last_signature, repeated_calls

    @staticmethod
    def _tool_signature(
        tool_name: str,
        raw_arguments: str,
    ) -> tuple[str, str]:
        try:
            arguments = json.loads(raw_arguments)
            normalized_arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            normalized_arguments = raw_arguments
        return tool_name, normalized_arguments


def serialize_tool_result(data: Any) -> str:
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False, default=str)
