from __future__ import annotations

import json

from ejagent.contracts.json import JsonValue, thaw_json_value
from ejagent.contracts.messages import (
    AssistantMessage,
    ConversationMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ejagent.contracts.runs import RunDelta, RunIntent, RunSpec


class _RunWorkspace:
    """Private mutable message workspace owned by one Kernel invocation."""

    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self._messages = list(spec.messages)
        self._delta: list[ConversationMessage] = []
        self._tool_call_ids: set[str] = set()
        self._last_tool_signature: tuple[str, str] | None = None
        self._repeated_tool_calls = 0
        self.turn = 0
        if spec.intent is RunIntent.TASK:
            assert spec.task is not None
            self.append(UserMessage(spec.task))

    @property
    def messages(self) -> tuple[ConversationMessage, ...]:
        return tuple(self._messages)

    @property
    def committed_messages(self) -> tuple[ConversationMessage, ...]:
        return self.spec.messages

    @property
    def pending_messages(self) -> tuple[ConversationMessage, ...]:
        return tuple(self._delta)

    @property
    def delta(self) -> RunDelta:
        return RunDelta(
            base_revision=self.spec.base_revision,
            messages=tuple(self._delta),
        )

    def advance_turn(self) -> int:
        self.turn += 1
        return self.turn

    def append(self, message: ConversationMessage) -> None:
        self._messages.append(message)
        self._delta.append(message)

    def append_assistant(self, message: AssistantMessage) -> None:
        for call in message.tool_calls:
            if call.id in self._tool_call_ids:
                raise ValueError(f"duplicate tool call id {call.id!r}")
            self._tool_call_ids.add(call.id)
        self.append(message)

    def append_tool_result(
        self,
        call: ToolCall,
        *,
        result: JsonValue,
        is_error: bool,
    ) -> ToolResultMessage:
        message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            result=result,
            is_error=is_error,
        )
        self.append(message)
        return message

    def discard_completion_claim(self, message: AssistantMessage) -> None:
        """Retain execution protocol, but omit rejected final prose from Conversation."""

        for messages in (self._messages, self._delta):
            for index in range(len(messages) - 1, -1, -1):
                if messages[index] is message:
                    if message.tool_calls:
                        messages[index] = AssistantMessage(
                            tool_calls=message.tool_calls
                        )
                    else:
                        messages.pop(index)
                    break

    def record_tool_call(self, call: ToolCall) -> bool:
        """Update the repeat guard and return whether its limit was reached."""

        signature = (
            call.name,
            json.dumps(
                thaw_json_value(call.arguments),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if signature == self._last_tool_signature:
            self._repeated_tool_calls += 1
        else:
            self._last_tool_signature = signature
            self._repeated_tool_calls = 1
        return self._repeated_tool_calls >= self.spec.limits.max_repeated_tool_calls
