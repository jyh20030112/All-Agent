from __future__ import annotations

from dataclasses import dataclass

from ejagent.contracts.messages import ConversationMessage, is_conversation_message


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    """Immutable committed Conversation at one logical revision."""

    revision: int = 0
    messages: tuple[ConversationMessage, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("conversation revision must be an integer")
        if self.revision < 0:
            raise ValueError("conversation revision must not be negative")
        object.__setattr__(self, "messages", tuple(self.messages))
        if not all(is_conversation_message(message) for message in self.messages):
            raise TypeError(
                "conversation messages must contain ConversationMessage values"
            )
