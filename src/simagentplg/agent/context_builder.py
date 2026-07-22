from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from simagentplg.agent.state import AgentState
from simagentplg.agent.types import INTERNAL_METADATA_PREFIX, AgentMessage
from simagentplg.handlers.definition import (
    ToolDefinition,
    ToolDefinitionInput,
    normalize_tool_definitions,
)
from simagentplg.plugins.skill.skill_manager import SkillManager


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """One complete model request built from agent state."""

    agent_messages: tuple[AgentMessage, ...]
    llm_messages: tuple[AgentMessage, ...]
    tools: tuple[dict[str, Any], ...]
    tool_definitions: tuple[ToolDefinition, ...] = ()


class AgentContextBuilder:
    """Build provider-ready context from agent state without mutating it."""

    def __init__(
        self,
        *,
        skill_manager: SkillManager | None = None,
    ) -> None:
        self._skill_manager = skill_manager

    def build(
        self,
        state: AgentState,
        *,
        tools: Sequence[ToolDefinitionInput] = (),
        transient_messages: Sequence[Mapping[str, Any]] = (),
    ) -> ContextBuildResult:
        """Build the context for one model request.

        Persistent conversation is copied from ``state``. Skill instructions,
        per-turn control messages, and tool definitions are combined into a
        complete provider request without mutating the state history.
        """

        context = self._copy_messages(state.messages)
        self._insert_skill_context(context, state.active_skill_name)
        context.extend(self._copy_messages(transient_messages))
        projected = self.convert_to_llm_messages(context)
        llm_messages = self._strip_internal_metadata(projected)
        definitions = normalize_tool_definitions(tools)
        return ContextBuildResult(
            agent_messages=tuple(context),
            llm_messages=tuple(llm_messages),
            tools=tuple(tool.to_openai_tool() for tool in definitions),
            tool_definitions=definitions,
        )

    def convert_to_llm_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[AgentMessage]:
        """Convert internal messages into provider-compatible messages.

        The default is copy-only. Subclasses can inspect internal metadata,
        filter records, or adapt custom message types without mutating
        ``AgentState``. Core-only metadata is removed after this hook returns.
        """

        return self._copy_messages(messages)

    @staticmethod
    def _strip_internal_metadata(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[AgentMessage]:
        return [
            {
                key: value
                for key, value in message.items()
                if key != "usage" and not key.startswith(INTERNAL_METADATA_PREFIX)
            }
            for message in messages
        ]

    def _insert_skill_context(
        self,
        context: list[AgentMessage],
        active_skill_name: str | None,
    ) -> None:
        if self._skill_manager is None:
            return

        skill_messages: list[AgentMessage] = []
        index_message = self._skill_manager.build_index_message()
        if index_message is not None:
            skill_messages.append(dict(index_message))
        if active_skill_name is not None:
            skill_messages.append(
                self._skill_manager.build_skill_context_message(active_skill_name)
            )
        if not skill_messages:
            return

        insert_at = self._system_message_end(context)
        context[insert_at:insert_at] = skill_messages

    @staticmethod
    def _copy_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[AgentMessage]:
        return [dict(message) for message in messages]

    @staticmethod
    def _system_message_end(
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        index = 0
        while index < len(messages) and messages[index].get("role") == "system":
            index += 1
        return index
