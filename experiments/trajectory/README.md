# Trajectory experiments

> PROTOTYPE — these files are experiment fixtures and disposable analysis
> tools. They are not EJAgent Runtime APIs.

This directory validates the trajectory concepts described in:

- [`../../docs/trajectory-failure-scenarios.md`](../../docs/trajectory-failure-scenarios.md)
- [`../../docs/fs-001-authentication-experiment.md`](../../docs/fs-001-authentication-experiment.md)
- [`../../docs/event-context-exposure.md`](../../docs/event-context-exposure.md)

The local experiment command is intentionally isolated from the production
package:

```bash
uv run python experiments/trajectory/run_experiments.py
```

It runs the FS-001 baseline, Gold Solution controls, deterministic period-two
failure replay through the current `RuntimeKernel`, and the four healthy
controls. It writes no durable application state.

The shareable state-model prototype is
[`trajectory_logic_prototype.html`](trajectory_logic_prototype.html); open it
directly in a browser.

The completed 2026-09-01 experiment is summarized in:

- [`../../docs/trajectory-experiment-report.md`](../../docs/trajectory-experiment-report.md)
- [`../../docs/trajectory-shadow-design.md`](../../docs/trajectory-shadow-design.md)
- [`../../docs/trajectory-context-projection.md`](../../docs/trajectory-context-projection.md)
- [`results/2026-09-01-summary.json`](results/2026-09-01-summary.json)
- [`results/2026-09-01-phase2-summary.json`](results/2026-09-01-phase2-summary.json)

The Phase-2 entry gates exercise a second failure domain, Fact validity,
concurrent causal attribution, the false-positive controls, and the event to
Context exposure matrix:

```bash
UV_CACHE_DIR=/tmp/ejagent-uv-cache uv run python \
  experiments/trajectory/phase2_evidence.py \
  --json-output /tmp/ejagent-trajectory-phase2.json
```

Existing generated live artifacts can be replayed through the observation-only
internal Analyzer without making Provider calls:

```bash
UV_CACHE_DIR=/tmp/ejagent-uv-cache uv run python \
  experiments/trajectory/replay_shadow.py \
  /tmp/ejagent-trajectory-live.json
```

Live Provider trials are opt-in and must use a pre-registered configuration.
Run `--help` for the current experimental options. Never commit credentials or
unredacted Provider payloads.

```bash
UV_CACHE_DIR=/tmp/ejagent-uv-cache uv run python \
  experiments/trajectory/run_experiments.py \
  --live \
  --json-output /tmp/ejagent-trajectory-local.json \
  --live-json-output /tmp/ejagent-trajectory-live.json
```
