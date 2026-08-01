from __future__ import annotations

from typing import Any

from ejagent.skills import Skill, SkillCatalog


class SkillManager(SkillCatalog):
    """Compatibility facade for the legacy dictionary-message API."""

    def build_index_message(self) -> dict[str, str] | None:
        content = self.build_index_content()
        return {"role": "system", "content": content} if content is not None else None

    def build_skill_context_message(self, skill_name: str) -> dict[str, str]:
        return {
            "role": "system",
            "content": self.build_skill_context_content(skill_name),
        }

    def select_explicit_skill(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        return self.select_explicit_skill_from_text(self._latest_user_task(messages))

    @staticmethod
    def _latest_user_task(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                return content if isinstance(content, str) else str(content)
        return ""


__all__ = ["Skill", "SkillManager"]
