"""Disposable ContextView projection implementations."""

from ejagent.context.pipeline import (
    DerivedCompactionPipeline,
    IdentityContextPipeline,
)
from ejagent.context.skills import SkillsContextPipeline

__all__ = [
    "DerivedCompactionPipeline",
    "IdentityContextPipeline",
    "SkillsContextPipeline",
]
