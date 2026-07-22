import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ejagent.logger import get_logger

logger = get_logger("skill")

_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    skill_md: Path
    description: str = ""
    template_md: Path | None = None
    sample_md: Path | None = None


class SkillManager:
    """Discover local skills and build explicit context projections."""

    def __init__(self, skills_root: str | Path):
        """
        Args:
            skills_root: Root directory whose child folders may contain SKILL.md.
        """
        self.skills_root = Path(skills_root)
        self._skills: dict[str, Skill] = {}
        self._discovered = False
        self._index_message: dict[str, str] | None = None
        self._skill_context_messages: dict[str, dict[str, str]] = {}
        logger.info("Skill registry initialized root=%s", self.skills_root)

    @property
    def skills(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    @property
    def discovered(self) -> bool:
        return self._discovered

    async def discover(self) -> None:
        """Scan child directories containing SKILL.md and build an index."""

        if self._discovered:
            return

        if not self.skills_root.exists():
            raise FileNotFoundError(f"skills root not found: {self.skills_root}")

        skills: dict[str, Skill] = {}

        for child in sorted(self.skills_root.iterdir()):
            if not child.is_dir():
                continue

            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue

            frontmatter = self._read_frontmatter(skill_md)
            name = self._skill_name(child, frontmatter)
            template_md = child / "template.md"
            sample_md = child / "examples" / "sample.md"

            if name in skills:
                raise ValueError(f"duplicate skill name: {name!r}")

            skills[name] = Skill(
                name=name,
                skill_md=skill_md,
                description=str(frontmatter.get("description", "")),
                template_md=template_md if template_md.exists() else None,
                sample_md=sample_md if sample_md.exists() else None,
            )

        self._skills = skills
        self._discovered = True
        self._index_message = None
        self._skill_context_messages.clear()

        if not self._skills:
            logger.debug(
                "Skill discovery completed with no skills root=%s", self.skills_root
            )
        else:
            logger.info(
                "Discovered %d skill(s): %s",
                len(self._skills),
                list(self._skills.keys()),
            )

    def build_index_message(self) -> dict[str, str] | None:
        """Return compact skill metadata for the model context."""

        if not self._skills:
            return None
        if self._index_message is not None:
            return dict(self._index_message)

        lines = [
            "Local skills are available. Use a skill when its description matches",
            "the user's task. An agent with a file-reading tool can read the full",
            "instructions from the listed location. If the user names a skill as",
            "$skill_name or skill:skill_name, its full instructions are injected",
            "into the current model context.",
            "",
            "Available skills:",
        ]
        for skill in self._skills.values():
            description = skill.description or "No description provided."
            lines.extend(
                [
                    f"- name: {skill.name}",
                    f"  description: {description}",
                    f"  location: {skill.skill_md.resolve()}",
                ]
            )
        self._index_message = {
            "role": "system",
            "content": "\n".join(lines),
        }
        return dict(self._index_message)

    def build_skill_context_message(
        self,
        skill_name: str,
    ) -> dict[str, str]:
        """Load full skill instructions for provider context."""

        if skill_name in self._skill_context_messages:
            return dict(self._skill_context_messages[skill_name])

        skill = self.get(skill_name)
        message = {
            "role": "system",
            "content": "\n".join(self._skill_content_parts(skill)),
        }
        self._skill_context_messages[skill_name] = message
        return dict(message)

    def get(self, skill_name: str) -> Skill:
        try:
            return self._skills[skill_name]
        except KeyError as exc:
            available = ", ".join(self._skills) or "none"
            raise KeyError(
                f"unknown skill {skill_name!r}; available skills: {available}"
            ) from exc

    def select_explicit_skill(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        """Return a locally named skill from the latest user message."""

        task = self._latest_user_task(messages)
        if not task:
            return None

        for name in self._skills:
            if re.search(rf"(?<!\w)\${re.escape(name)}(?!\w)", task):
                return name
            if re.search(
                rf"(?<!\w)skill:{re.escape(name)}(?!\w)",
                task,
                flags=re.IGNORECASE,
            ):
                return name
        return None

    @staticmethod
    def _read_frontmatter(skill_md: Path) -> dict[str, Any]:
        text = skill_md.read_text(encoding="utf-8")
        match = _FRONTMATTER_PATTERN.match(text)
        if match is None:
            return {}

        data = yaml.safe_load(match.group(1)) or {}
        if not isinstance(data, dict):
            raise ValueError(f"SKILL.md frontmatter must be a mapping: {skill_md}")
        return dict(data)

    @staticmethod
    def _skill_name(
        skill_dir: Path,
        frontmatter: dict[str, Any],
    ) -> str:
        raw_name = frontmatter.get("name") or skill_dir.name
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"skill name must be a non-empty string: {skill_dir}")
        return raw_name.strip()

    @staticmethod
    def _latest_user_task(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                return content if isinstance(content, str) else str(content)
        return ""

    @staticmethod
    def _skill_content_parts(skill: Skill) -> list[str]:
        parts = [
            f'You are executing the local skill "{skill.name}".',
            "",
            "[SKILL.md]",
            skill.skill_md.read_text(encoding="utf-8").strip(),
        ]

        if skill.template_md is not None:
            parts.extend(
                [
                    "",
                    "[template.md]",
                    skill.template_md.read_text(encoding="utf-8").strip(),
                ]
            )

        if skill.sample_md is not None:
            parts.extend(
                [
                    "",
                    "[examples/sample.md]",
                    skill.sample_md.read_text(encoding="utf-8").strip(),
                ]
            )
        return parts
