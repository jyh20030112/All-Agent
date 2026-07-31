"""Disposable ContextView projection implementations."""

from ejagent.context.pipeline import (
    DerivedCompactionPipeline,
    IdentityContextPipeline,
)

__all__ = ["DerivedCompactionPipeline", "IdentityContextPipeline"]
