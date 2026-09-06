---
status: accepted
---

# The Harness delegates trajectory observation boundaries to the Kernel

`RuntimeKernel` defines a small, optional `TrajectoryMonitor` seam in
`ejagent.kernel.trajectory`. The Kernel emits observation signals at the Run
baseline, after a complete Tool batch has been committed to the private
workspace, and before accepting a text Completion Claim. Internal
`ejagent._trajectory` code implements that seam and may stage evaluator output
through the independent `ContextPipeline` seam.

This keeps the dependency direction stable: the Kernel identifies when an observation
is causally meaningful, while the host evaluator decides what current
environment truth means. The Kernel does not import cycle detection, Fact models,
or Context projection internals. `AgentHarness` accepts the monitor as a
composition dependency and passes it to the Kernel; the host connects its
evaluator and Context policy. These components together provide the Harness's
trajectory feedback capability.

The first integration is observation-only and fail-open. A monitor failure is
audited, disables further capture for that Run, and cannot change its result.
A failed Completion Audit is recorded but does not yet continue or reject the
Run. Monitor state is closed on every Kernel exit, including escaping protocol
errors.

The rejected alternatives were a direct Kernel dependency on
`ejagent._trajectory`, which would invert the intended architecture, and a
default-on controller, which would grant enforcement authority before execution
telemetry establishes acceptable false-positive and availability behavior.
