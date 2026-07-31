from __future__ import annotations

from ejagent.contracts.context import (
    ContextBuildError,
    ContextCompactionOutput,
    ContextCompactionRequest,
    ContextCompactor,
    ContextCompactorError,
    ContextPipeline,
    ContextProtocolError,
    ContextRequest,
    ContextView,
)
from ejagent.contracts.control import CancellationToken, RunCancelledError
from ejagent.contracts.lifecycle import ManagedResource
from ejagent.contracts.messages import ContextSummary, SystemMessage
from ejagent.contracts.runs import FailureCode


class IdentityContextPipeline(ContextPipeline):
    """Project committed and pending messages without transformation."""

    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        cancellation.raise_if_cancelled()
        return ContextView(
            run_id=request.run_id,
            source_revision=request.source_revision,
            turn=request.turn,
            messages=request.messages,
            metadata={**request.metadata, "projection": "identity"},
        )


class DerivedCompactionPipeline(ContextPipeline):
    """Replace committed non-system history with a disposable summary view."""

    def __init__(
        self,
        compactor: ContextCompactor,
        *,
        minimum_messages: int = 20,
    ) -> None:
        if isinstance(minimum_messages, bool) or not isinstance(minimum_messages, int):
            raise TypeError("minimum_messages must be an integer")
        if minimum_messages <= 0:
            raise ValueError("minimum_messages must be greater than zero")
        self._compactor = compactor
        self._minimum_messages = minimum_messages

    async def start(self) -> None:
        if isinstance(self._compactor, ManagedResource):
            await self._compactor.start()

    async def shutdown(self) -> None:
        if isinstance(self._compactor, ManagedResource):
            await self._compactor.shutdown()

    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        cancellation.raise_if_cancelled()
        system_end = 0
        while system_end < len(request.committed_messages) and isinstance(
            request.committed_messages[system_end], SystemMessage
        ):
            system_end += 1
        stable_instructions = request.committed_messages[:system_end]
        history = request.committed_messages[system_end:]
        if len(history) < self._minimum_messages:
            return ContextView(
                run_id=request.run_id,
                source_revision=request.source_revision,
                turn=request.turn,
                messages=request.messages,
                metadata={**request.metadata, "projection": "identity"},
            )

        revision_start = 1 if request.source_revision > 0 else 0
        compaction_request = ContextCompactionRequest(
            messages=history,
            source_revision_start=revision_start,
            source_revision_end=request.source_revision,
        )
        try:
            output = await cancellation.run(
                self._compactor.compact(
                    compaction_request,
                    cancellation=cancellation,
                )
            )
        except RunCancelledError:
            raise
        except ContextCompactorError as exc:
            raise ContextBuildError(
                FailureCode.COMPACTION_FAILED,
                str(exc),
                retryable=exc.retryable,
            ) from exc
        except Exception as exc:
            raise ContextProtocolError(
                f"ContextCompactor raised an undeclared {type(exc).__name__}"
            ) from exc
        if not isinstance(output, ContextCompactionOutput):
            raise ContextProtocolError(
                "ContextCompactor.compact() must return ContextCompactionOutput"
            )

        summary = ContextSummary(
            source_revision_start=revision_start,
            source_revision_end=request.source_revision,
            content=output.content.strip(),
            compactor_id=output.compactor_id.strip(),
        )
        return ContextView(
            run_id=request.run_id,
            source_revision=request.source_revision,
            turn=request.turn,
            messages=(*stable_instructions, summary, *request.pending_messages),
            metadata={
                **request.metadata,
                "projection": "derived_compaction",
                "summarized_message_count": len(history),
            },
        )
