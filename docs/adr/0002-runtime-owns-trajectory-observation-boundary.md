---
status: accepted
---

# Runtime owns the trajectory observation boundary

`RuntimeKernel` defines a small, optional `TrajectoryMonitor` seam in
`ejagent.kernel.trajectory`. The Kernel emits Runtime-owned signals at the Run
baseline, after a complete Tool batch has been committed to the private
workspace, and before accepting a text Completion Claim. Internal
`ejagent._trajectory` code implements that seam and may stage evaluator output
through the independent `ContextPipeline` seam.

This keeps the dependency direction stable: Runtime names when an observation
is causally meaningful, while the host evaluator decides what current
environment truth means. Runtime does not import cycle detection, Fact models,
or Context projection internals. `AgentHarness` only passes the optional seam
through to its Kernel.

The first integration is observation-only and fail-open. A monitor failure is
audited, disables further capture for that Run, and cannot change its result.
A failed Completion Audit is recorded but does not yet continue or reject the
Run. Monitor state is closed on every Runtime exit, including escaping protocol
errors.

The rejected alternatives were a direct Kernel dependency on
`ejagent._trajectory`, which would invert the intended architecture, and a
default-on controller, which would grant enforcement authority before runtime
telemetry establishes acceptable false-positive and availability behavior.
