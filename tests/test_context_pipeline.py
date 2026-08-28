from __future__ import annotations

import unittest
from collections.abc import AsyncIterator, Sequence

from ejagent.context import DerivedCompactionPipeline, IdentityContextPipeline
from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    CancellationToken,
    ContextBuildError,
    ContextCompactionOutput,
    ContextCompactionRequest,
    ContextCompactor,
    ContextCompactorError,
    ContextPipeline,
    ContextProtocolError,
    ContextRequest,
    ContextSummary,
    ContextView,
    FailureCode,
    ModelPort,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    RunIntent,
    RunPhase,
    RunSpec,
    RunStatus,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutor,
    TransientInstruction,
    UserMessage,
)
from ejagent.harness import AgentHarness
from ejagent.kernel import RuntimeKernel


def context_request(
    *,
    committed: tuple[SystemMessage | UserMessage | AssistantMessage, ...] = (),
    pending: tuple[SystemMessage | UserMessage | AssistantMessage, ...] = (),
    revision: int = 3,
    turn: int = 1,
) -> ContextRequest:
    return ContextRequest(
        run_id="context-run",
        source_revision=revision,
        turn=turn,
        committed_messages=committed,
        pending_messages=pending,
        metadata={"tenant": "test"},
    )


class StaticCompactor(ContextCompactor):
    def __init__(self, content: str = "earlier work") -> None:
        self.content = content
        self.requests: list[ContextCompactionRequest] = []

    async def compact(
        self,
        request: ContextCompactionRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextCompactionOutput:
        self.requests.append(request)
        return ContextCompactionOutput(self.content, "static-compactor")


class FailingCompactor(ContextCompactor):
    async def compact(
        self,
        request: ContextCompactionRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextCompactionOutput:
        raise ContextCompactorError("summary backend unavailable", retryable=True)


class BrokenCompactor(ContextCompactor):
    async def compact(
        self,
        request: ContextCompactionRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextCompactionOutput:
        raise ValueError("adapter leaked implementation failure")


class RecordingModel(ModelPort):
    def __init__(self, responses: Sequence[AssistantMessage]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelResponseCompleted(self.responses.pop(0))


class NoTools(ToolExecutor):
    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return ()

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        raise AssertionError("no tools are registered")


class SteeringProjection(ContextPipeline):
    def __init__(self) -> None:
        self.requests: list[ContextRequest] = []

    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        self.requests.append(request)
        return ContextView(
            run_id=request.run_id,
            source_revision=request.source_revision,
            turn=request.turn,
            messages=(
                *request.messages,
                TransientInstruction("change direction", "test-steering"),
            ),
        )


class FailingPipeline(ContextPipeline):
    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        raise ContextBuildError(
            FailureCode.COMPACTION_FAILED,
            "cannot summarize",
            retryable=True,
        )


class ContextPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_projection_preserves_messages_without_aliasing(
        self,
    ) -> None:
        committed = (SystemMessage("stable"), UserMessage("old"))
        pending = (UserMessage("new"),)
        request = context_request(committed=committed, pending=pending)

        view = await IdentityContextPipeline().build(
            request,
            cancellation=CancellationSource().token,
        )

        self.assertEqual(view.messages, (*committed, *pending))
        self.assertEqual(view.source_revision, 3)
        self.assertEqual(view.metadata["projection"], "identity")
        self.assertEqual(request.committed_messages, committed)
        self.assertEqual(request.pending_messages, pending)

    async def test_derived_compaction_only_changes_the_context_view(self) -> None:
        committed = (
            SystemMessage("stable instruction"),
            UserMessage("old task"),
            AssistantMessage(content="old answer"),
        )
        pending = (UserMessage("current task"),)
        request = context_request(committed=committed, pending=pending)
        compactor = StaticCompactor()
        pipeline = DerivedCompactionPipeline(compactor, minimum_messages=2)

        view = await pipeline.build(
            request,
            cancellation=CancellationSource().token,
        )

        self.assertEqual(view.messages[0], committed[0])
        self.assertIsInstance(view.messages[1], ContextSummary)
        summary = view.messages[1]
        assert isinstance(summary, ContextSummary)
        self.assertEqual(summary.source_revision_start, 1)
        self.assertEqual(summary.source_revision_end, 3)
        self.assertEqual(summary.content, "earlier work")
        self.assertEqual(view.messages[2], pending[0])
        self.assertEqual(compactor.requests[0].messages, committed[1:])
        self.assertEqual(request.committed_messages, committed)
        self.assertNotIn(summary, request.messages)

    async def test_compaction_projection_is_rebuildable_from_same_history(self) -> None:
        request = context_request(
            committed=(
                UserMessage("one"),
                AssistantMessage(content="two"),
            )
        )
        pipeline = DerivedCompactionPipeline(
            StaticCompactor("deterministic summary"),
            minimum_messages=2,
        )

        first = await pipeline.build(
            request,
            cancellation=CancellationSource().token,
        )
        second = await pipeline.build(
            request,
            cancellation=CancellationSource().token,
        )

        self.assertEqual(first, second)
        self.assertEqual(request.messages[0], UserMessage("one"))

    async def test_below_threshold_uses_unmodified_projection(self) -> None:
        compactor = StaticCompactor()
        request = context_request(committed=(UserMessage("only one"),))

        view = await DerivedCompactionPipeline(
            compactor,
            minimum_messages=2,
        ).build(request, cancellation=CancellationSource().token)

        self.assertEqual(view.messages, request.messages)
        self.assertEqual(view.metadata["projection"], "identity")
        self.assertEqual(compactor.requests, [])

    async def test_declared_compactor_failure_becomes_context_failure(self) -> None:
        pipeline = DerivedCompactionPipeline(
            FailingCompactor(),
            minimum_messages=1,
        )

        with self.assertRaises(ContextBuildError) as raised:
            await pipeline.build(
                context_request(committed=(UserMessage("old"),)),
                cancellation=CancellationSource().token,
            )

        self.assertEqual(raised.exception.code, FailureCode.COMPACTION_FAILED)
        self.assertTrue(raised.exception.retryable)

    async def test_undeclared_compactor_failure_is_a_protocol_error(self) -> None:
        pipeline = DerivedCompactionPipeline(
            BrokenCompactor(),
            minimum_messages=1,
        )

        with self.assertRaises(ContextProtocolError):
            await pipeline.build(
                context_request(committed=(UserMessage("old"),)),
                cancellation=CancellationSource().token,
            )

    async def test_kernel_projects_transient_context_without_committing_it(
        self,
    ) -> None:
        model = RecordingModel([AssistantMessage(content="done")])
        context = SteeringProjection()
        spec = RunSpec(
            run_id="transient-run",
            base_revision=2,
            intent=RunIntent.TASK,
            task="original task",
            messages=(SystemMessage("stable"),),
        )

        outcome = await RuntimeKernel(
            model=model,
            tools=NoTools(),
            context=context,
        ).run(spec)

        self.assertIsInstance(model.requests[0].messages[-1], TransientInstruction)
        self.assertNotIn(
            TransientInstruction("change direction", "test-steering"),
            outcome.delta.messages,
        )
        self.assertEqual(
            context.requests[0].pending_messages, (UserMessage("original task"),)
        )
        self.assertIn(
            "context_built", [record.kind for record in outcome.audit_records]
        )

    async def test_context_failure_is_a_structured_kernel_outcome(self) -> None:
        spec = RunSpec(
            run_id="context-failure",
            base_revision=0,
            intent=RunIntent.TASK,
            task="task",
            messages=(),
        )
        kernel = RuntimeKernel(
            model=RecordingModel([]),
            tools=NoTools(),
            context=FailingPipeline(),
        )

        outcome = await kernel.run(spec)

        self.assertEqual(outcome.result.status, RunStatus.FAILED)
        self.assertEqual(outcome.result.stop_reason, StopReason.COMPACTION_FAILED)
        assert outcome.failure is not None
        self.assertEqual(outcome.failure.phase, RunPhase.CONTEXT)
        self.assertEqual(outcome.failure.code, FailureCode.COMPACTION_FAILED)
        self.assertEqual(outcome.delta.messages, (UserMessage("task"),))

    async def test_harness_compaction_never_rewrites_conversation(self) -> None:
        model = RecordingModel(
            [
                AssistantMessage(content="first answer"),
                AssistantMessage(content="second answer"),
            ]
        )
        run_ids = iter(("first-run", "second-run"))
        harness = AgentHarness(
            agent_id="derived-context-agent",
            model=model,
            tools=NoTools(),
            context=DerivedCompactionPipeline(
                StaticCompactor("first exchange summarized"),
                minimum_messages=2,
            ),
            initial_messages=(SystemMessage("stable"),),
            run_id_factory=lambda: next(run_ids),
        )

        await harness.run("first task")
        before_second = harness.messages
        await harness.run("second task")

        self.assertIsInstance(model.requests[1].messages[1], ContextSummary)
        self.assertEqual(model.requests[1].messages[-1], UserMessage("second task"))
        self.assertEqual(harness.messages[: len(before_second)], before_second)
        self.assertFalse(
            any(isinstance(message, ContextSummary) for message in harness.messages)
        )


if __name__ == "__main__":
    unittest.main()
