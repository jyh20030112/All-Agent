from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ejagent.contracts.audit import RunAudit
from ejagent.contracts.conversation import ConversationSnapshot
from ejagent.contracts.json import freeze_json_value
from ejagent.contracts.messages import (
    AssistantMessage,
    ConversationMessage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ejagent.contracts.runs import (
    FailureCode,
    RunFailure,
    RunPhase,
    RunResult,
    RunStatus,
)
from ejagent.contracts.session import SessionMigrationError, SessionSnapshot
from ejagent.session.types import AgentSession

_REMEDIATION = (
    "export the legacy Session with only text, function tool calls, and "
    "finished Runs, then retry migration"
)


@dataclass(frozen=True, slots=True)
class LegacySessionMigration:
    """Typed recovery and Audit values decoded from one legacy Session."""

    source_session_id: str
    snapshot: SessionSnapshot
    audit: tuple[RunAudit, ...]


def migrate_legacy_session(
    session: AgentSession,
    *,
    agent_id: str | None = None,
) -> LegacySessionMigration:
    """Convert a detached legacy AgentSession without using compacted views."""

    target_agent_id = agent_id or session.agent_id
    if target_agent_id is None:
        raise SessionMigrationError(
            f"legacy Session {session.session_id!r} has no agent_id",
            remediation="supply agent_id explicitly when importing the Session",
        )
    target_agent_id = target_agent_id.strip()
    if not target_agent_id:
        raise SessionMigrationError(
            "migration agent_id is empty",
            remediation="supply a non-empty agent_id",
        )
    if session.agent_id is not None and session.agent_id != target_agent_id:
        raise SessionMigrationError(
            f"legacy Session belongs to agent {session.agent_id!r}, not "
            f"{target_agent_id!r}",
            remediation="use the original agent_id or migrate into a new store key",
        )

    unfinished = [run.run_id for run in session.runs if not run.finished]
    if unfinished:
        raise SessionMigrationError(
            "legacy Session contains unfinished Run(s): " + ", ".join(unfinished),
            remediation="finish or explicitly discard those Runs before migration",
        )

    call_names: dict[str, str] = {}
    messages: list[ConversationMessage] = []
    for index, entry in enumerate(session.entries):
        try:
            messages.append(_legacy_message(entry.message, call_names=call_names))
        except SessionMigrationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionMigrationError(
                f"legacy message {index} cannot be represented: {exc}",
                remediation=_REMEDIATION,
            ) from exc

    revision = 0
    audit: list[RunAudit] = []
    last_result: RunResult | None = None
    for run in session.runs:
        assert run.result is not None
        result = RunResult(
            run_id=run.run_id,
            status=run.result.status,
            stop_reason=run.result.stop_reason,
            turns=run.result.turns,
            output=run.result.output,
            usage=run.result.usage,
        )
        committed = result.status is RunStatus.COMPLETED
        failure = (
            RunFailure(
                phase=RunPhase.RUNTIME,
                code=FailureCode.RUNTIME_ERROR,
                message=run.result.error or result.stop_reason.value,
            )
            if result.status is RunStatus.FAILED
            else None
        )
        next_revision = revision + int(committed)
        audit.append(
            RunAudit(
                result=result,
                base_revision=revision,
                resulting_revision=next_revision,
                committed=committed,
                failure=failure,
            )
        )
        revision = next_revision
        if committed:
            last_result = result

    return LegacySessionMigration(
        source_session_id=session.session_id,
        snapshot=SessionSnapshot(
            agent_id=target_agent_id,
            conversation=ConversationSnapshot(
                revision=revision,
                messages=tuple(messages),
            ),
            last_result=last_result,
        ),
        audit=tuple(audit),
    )


def _legacy_message(
    message: Mapping[str, Any],
    *,
    call_names: dict[str, str],
) -> ConversationMessage:
    role = _required_string(message.get("role"), "message.role")
    if role == "system":
        return SystemMessage(_required_string(message.get("content"), "content"))
    if role == "user":
        return UserMessage(_required_string(message.get("content"), "content"))
    if role == "assistant":
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise TypeError("assistant content must be text or null")
        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise TypeError("assistant tool_calls must be an array")
        calls = tuple(
            _legacy_tool_call(value, index=index)
            for index, value in enumerate(raw_calls)
        )
        for call in calls:
            call_names[call.id] = call.name
        return AssistantMessage(content=content, tool_calls=calls)
    if role == "tool":
        call_id = _required_string(message.get("tool_call_id"), "tool_call_id")
        raw_name = message.get("name")
        if raw_name is None:
            name = call_names.get(call_id)
            if name is None:
                raise ValueError(
                    f"tool result {call_id!r} has no name and no preceding call"
                )
        else:
            name = _required_string(raw_name, "tool name")
        content = freeze_json_value(message.get("content"), label="tool content")
        raw_error = message.get("is_error", False)
        if not isinstance(raw_error, bool):
            raise TypeError("tool is_error must be a boolean")
        return ToolResultMessage(
            tool_call_id=call_id,
            tool_name=name,
            result=content,
            is_error=raw_error,
        )
    raise SessionMigrationError(
        f"legacy message role {role!r} is unsupported",
        remediation=_REMEDIATION,
    )


def _legacy_tool_call(value: Any, *, index: int) -> ToolCall:
    if not isinstance(value, Mapping):
        raise TypeError(f"tool_calls[{index}] must be an object")
    call_id = _required_string(value.get("id"), f"tool_calls[{index}].id")
    function = value.get("function")
    if function is not None:
        if not isinstance(function, Mapping):
            raise TypeError(f"tool_calls[{index}].function must be an object")
        name = _required_string(
            function.get("name"),
            f"tool_calls[{index}].function.name",
        )
        raw_arguments = function.get("arguments", "{}")
    else:
        name = _required_string(value.get("name"), f"tool_calls[{index}].name")
        raw_arguments = value.get("arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"tool_calls[{index}] arguments contain invalid JSON"
            ) from exc
    else:
        arguments = raw_arguments
    if not isinstance(arguments, Mapping):
        raise TypeError(f"tool_calls[{index}] arguments must decode to an object")
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value
