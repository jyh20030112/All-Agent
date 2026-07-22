import unittest

from ejagent import (
    AgentContextBuilder,
    AgentState,
    AssistantMessage,
    CancellationSource,
    CancellationToken,
    CompactionRequest,
    ContextBuildResult,
    ModelAdapter,
    ModelCompactor,
    prepare_compaction,
)


class SummaryModel(ModelAdapter):
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.contexts: list[ContextBuildResult] = []
        self.cancellations: list[CancellationToken | None] = []

    async def complete(
        self,
        context: ContextBuildResult,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AssistantMessage:
        self.contexts.append(context)
        self.cancellations.append(cancellation)
        return AssistantMessage(content=self.content)


def request() -> CompactionRequest:
    preparation = prepare_compaction(
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "recent"},
        ],
        keep_recent_tokens=1,
    )
    return CompactionRequest(preparation)


class ModelCompactorTests(unittest.IsolatedAsyncioTestCase):
    async def test_injected_builder_and_model_produce_compactor_output(self) -> None:
        model = SummaryModel("  concise summary  ")
        built_requests: list[CompactionRequest] = []

        def build_context(compaction: CompactionRequest) -> ContextBuildResult:
            built_requests.append(compaction)
            state = AgentState(
                messages=[
                    {"role": "system", "content": "Summarize reliably."},
                    {
                        "role": "user",
                        "content": repr(compaction.preparation.messages_to_summarize),
                    },
                ]
            )
            return AgentContextBuilder().build(state)

        compactor = ModelCompactor(
            model,
            context_builder=build_context,
            source="summary-model:test",
        )
        source = CancellationSource()
        active_request = request()

        output = await compactor.compact(
            active_request,
            cancellation=source.token,
        )

        self.assertEqual(output.content, "concise summary")
        self.assertEqual(output.source, "summary-model:test")
        self.assertEqual(built_requests, [active_request])
        self.assertEqual(len(model.contexts), 1)
        self.assertIs(model.cancellations[0], source.token)

    async def test_empty_model_response_is_rejected(self) -> None:
        model = SummaryModel(None)
        context = AgentContextBuilder().build(AgentState())
        compactor = ModelCompactor(
            model,
            context_builder=lambda _: context,
            source="summary-model:test",
        )

        with self.assertRaisesRegex(RuntimeError, "empty content"):
            await compactor.compact(request())

    async def test_invalid_context_builder_result_is_rejected(self) -> None:
        model = SummaryModel("summary")
        compactor = ModelCompactor(
            model,
            context_builder=lambda _: object(),  # type: ignore[arg-type,return-value]
            source="summary-model:test",
        )

        with self.assertRaisesRegex(TypeError, "ContextBuildResult"):
            await compactor.compact(request())
        self.assertEqual(model.contexts, [])

    async def test_cancelled_request_does_not_call_model(self) -> None:
        model = SummaryModel("summary")
        context = AgentContextBuilder().build(AgentState())
        compactor = ModelCompactor(
            model,
            context_builder=lambda _: context,
            source="summary-model:test",
        )
        source = CancellationSource()
        source.cancel("stop summary")

        with self.assertRaisesRegex(RuntimeError, "stop summary"):
            await compactor.compact(request(), cancellation=source.token)
        self.assertEqual(model.contexts, [])

    def test_source_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "source"):
            ModelCompactor(
                SummaryModel("summary"),
                context_builder=lambda _: AgentContextBuilder().build(AgentState()),
                source=" ",
            )


if __name__ == "__main__":
    unittest.main()
