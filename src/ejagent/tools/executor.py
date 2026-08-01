from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.lifecycle import ManagedResource
from ejagent.contracts.messages import ToolCall
from ejagent.contracts.tools import (
    ToolDefinition,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutor,
    ToolProtocolError,
)

ToolFunction = Callable[
    [ToolCall, CancellationToken],
    Awaitable[ToolExecutionResult],
]


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Pair one provider-neutral definition with its async implementation."""

    definition: ToolDefinition
    function: ToolFunction

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("function tool definition must be a ToolDefinition")
        if not callable(self.function):
            raise TypeError("function tool implementation must be callable")


class FunctionToolExecutor(ToolExecutor):
    """Validate and dispatch a fixed set of in-process async tools."""

    def __init__(self, tools: Iterable[FunctionTool] = ()) -> None:
        items = tuple(tools)
        if not all(isinstance(item, FunctionTool) for item in items):
            raise TypeError("tools must contain FunctionTool values")
        names = [item.definition.name for item in items]
        if len(names) != len(set(names)):
            raise ValueError("function tools must have unique names")
        self._tools = {item.definition.name: item for item in items}
        self._definitions = tuple(item.definition for item in items)

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return self._definitions

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        try:
            tool = self._tools[call.name]
        except KeyError as exc:
            raise ToolExecutionError(f"unknown function tool {call.name!r}") from exc
        result = await tool.function(call, cancellation)
        if not isinstance(result, ToolExecutionResult):
            raise ToolProtocolError(
                f"function tool {call.name!r} returned "
                f"{type(result).__name__}, expected ToolExecutionResult"
            )
        return result


class CompositeToolExecutor(ToolExecutor):
    """Present multiple ToolExecutors as one lifecycle-managed namespace."""

    def __init__(self, executors: Iterable[ToolExecutor]) -> None:
        self._executors = tuple(executors)
        if not self._executors:
            raise ValueError("executors must not be empty")
        self._routes: dict[str, ToolExecutor] = {}
        self._definitions: tuple[ToolDefinition, ...] = ()
        self._started: tuple[ManagedResource, ...] = ()
        self._is_started = False
        self._refresh_routes()

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return self._definitions

    async def start(self) -> None:
        if self._is_started:
            return
        started: list[ManagedResource] = []
        try:
            for executor in self._executors:
                if isinstance(executor, ManagedResource):
                    await executor.start()
                    started.append(executor)
            self._refresh_routes()
        except BaseException:
            for resource in reversed(started):
                try:
                    await resource.shutdown()
                except BaseException:
                    pass
            raise
        self._started = tuple(started)
        self._is_started = True

    async def shutdown(self) -> None:
        errors: list[Exception] = []
        for resource in reversed(self._started):
            try:
                await resource.shutdown()
            except Exception as exc:
                errors.append(exc)
        self._started = ()
        self._is_started = False
        self._refresh_routes()
        if errors:
            raise ExceptionGroup("ToolExecutor shutdown failed", errors)

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        try:
            executor = self._routes[call.name]
        except KeyError as exc:
            raise ToolExecutionError(f"unknown composed tool {call.name!r}") from exc
        return await executor.execute(call, cancellation=cancellation)

    def _refresh_routes(self) -> None:
        routes: dict[str, ToolExecutor] = {}
        definitions: list[ToolDefinition] = []
        for executor in self._executors:
            for definition in executor.definitions:
                if not isinstance(definition, ToolDefinition):
                    raise ToolProtocolError(
                        "composed executor exposed a non-ToolDefinition value"
                    )
                if definition.name in routes:
                    raise ToolProtocolError(
                        f"duplicate composed tool name {definition.name!r}"
                    )
                routes[definition.name] = executor
                definitions.append(definition)
        self._routes = routes
        self._definitions = tuple(definitions)
