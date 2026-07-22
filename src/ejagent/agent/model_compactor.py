from __future__ import annotations

from typing import Protocol

from ejagent.agent.cancellation import CancellationToken
from ejagent.agent.compaction import (
    CompactionRequest,
    CompactorOutput,
)
from ejagent.agent.context_builder import ContextBuildResult
from ejagent.providers.base import AssistantMessage, ModelAdapter


class CompactionContextBuilder(Protocol):
    """Build one provider request from a prepared compaction operation."""

    def __call__(self, request: CompactionRequest) -> ContextBuildResult:
        """Return the model context, including the application-owned prompt."""


class ModelCompactor:
    """Adapt a borrowed ``ModelAdapter`` into the ``Compactor`` protocol.

    The caller owns the model lifecycle and supplies all prompt construction.
    This keeps the Core provider-neutral and avoids embedding a summary policy.
    """

    def __init__(
        self,
        model: ModelAdapter,
        *,
        context_builder: CompactionContextBuilder,
        source: str,
    ) -> None:
        source = source.strip()
        if not source:
            raise ValueError("model compactor source must not be empty")
        self.model = model
        self.context_builder = context_builder
        self.source = source

    async def compact(
        self,
        request: CompactionRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> CompactorOutput:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        context = self.context_builder(request)
        if not isinstance(context, ContextBuildResult):
            raise TypeError("CompactionContextBuilder must return ContextBuildResult")
        response = await self.model.complete(
            context,
            cancellation=cancellation,
        )
        if not isinstance(response, AssistantMessage):
            raise TypeError("ModelAdapter.complete() must return AssistantMessage")
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        content = response.content.strip() if response.content is not None else ""
        if not content:
            raise RuntimeError("compaction model returned empty content")
        return CompactorOutput(content=content, source=self.source)
