from __future__ import annotations

from pathlib import Path

from ejagent.context.pipeline import IdentityContextPipeline
from ejagent.contracts.context import (
    ContextPipeline,
    ContextProtocolError,
    ContextRequest,
    ContextView,
)
from ejagent.contracts.control import CancellationToken
from ejagent.contracts.lifecycle import ManagedResource
from ejagent.contracts.messages import TransientInstruction, UserMessage
from ejagent.skills import SkillCatalog


class SkillsContextPipeline(ContextPipeline):
    """Decorate ContextViews with a local skill index and explicit instructions."""

    def __init__(
        self,
        skills_root: str | Path,
        *,
        base: ContextPipeline | None = None,
    ) -> None:
        self._catalog = SkillCatalog(skills_root)
        self._base = base or IdentityContextPipeline()
        self._started = False

    @property
    def catalog(self) -> SkillCatalog:
        return self._catalog

    async def start(self) -> None:
        if self._started:
            return
        await self._catalog.discover()
        if isinstance(self._base, ManagedResource):
            await self._base.start()
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        try:
            if isinstance(self._base, ManagedResource):
                await self._base.shutdown()
        finally:
            self._started = False

    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        if not self._started:
            raise ContextProtocolError("SkillsContextPipeline is not started")
        cancellation.raise_if_cancelled()
        instructions: list[TransientInstruction] = []
        index = self._catalog.build_index_content()
        if index is not None:
            instructions.append(TransientInstruction(index, "skills:index"))
        task = self._latest_user_task(request)
        selected = self._catalog.select_explicit_skill_from_text(task)
        if selected is not None:
            instructions.append(
                TransientInstruction(
                    self._catalog.build_skill_context_content(selected),
                    f"skills:{selected}",
                )
            )
        augmented = ContextRequest(
            run_id=request.run_id,
            source_revision=request.source_revision,
            turn=request.turn,
            committed_messages=request.committed_messages,
            pending_messages=request.pending_messages,
            transient_instructions=(
                *instructions,
                *request.transient_instructions,
            ),
            metadata=request.metadata,
        )
        view = await cancellation.run(
            self._base.build(augmented, cancellation=cancellation)
        )
        if not isinstance(view, ContextView):
            raise ContextProtocolError(
                "wrapped ContextPipeline.build() must return ContextView"
            )
        return ContextView(
            run_id=view.run_id,
            source_revision=view.source_revision,
            turn=view.turn,
            messages=view.messages,
            metadata={
                **view.metadata,
                "available_skills": tuple(skill.name for skill in self._catalog.skills),
                "selected_skill": selected,
            },
        )

    @staticmethod
    def _latest_user_task(request: ContextRequest) -> str:
        for message in reversed(
            (*request.committed_messages, *request.pending_messages)
        ):
            if isinstance(message, UserMessage):
                return message.content
        return ""
