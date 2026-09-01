"""Internal trajectory analysis and opt-in Context; not a stable public Interface."""

from ejagent._trajectory.context import (
    ProjectedTrajectoryContext,
    TrajectoryContextEvent,
    TrajectoryContextEventKind,
    TrajectoryContextFrame,
    TrajectoryContextPipeline,
    TrajectoryContextProjector,
    immutable_frames,
)
from ejagent._trajectory.shadow import (
    EnvironmentFact,
    FactValidity,
    NormalizedAction,
    NormalizedObservation,
    ProgressSnapshot,
    ShadowTrajectoryAnalyzer,
    ShadowTrajectoryObserver,
    TrajectoryCheckpoint,
    TrajectoryReport,
    TrajectoryVerdict,
)

__all__ = [
    "EnvironmentFact",
    "FactValidity",
    "NormalizedAction",
    "NormalizedObservation",
    "ProjectedTrajectoryContext",
    "ProgressSnapshot",
    "ShadowTrajectoryAnalyzer",
    "ShadowTrajectoryObserver",
    "TrajectoryCheckpoint",
    "TrajectoryContextEvent",
    "TrajectoryContextEventKind",
    "TrajectoryContextFrame",
    "TrajectoryContextPipeline",
    "TrajectoryContextProjector",
    "TrajectoryReport",
    "TrajectoryVerdict",
    "immutable_frames",
]
