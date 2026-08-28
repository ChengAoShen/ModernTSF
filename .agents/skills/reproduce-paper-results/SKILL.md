---
name: reproduce-paper-results
description: Reproduce and compare a forecasting paper's reported experiments in ModernTSF by mapping its protocol to runnable configs and aligned metrics. Use for paper-result replication; not for implementing the model or designing an unrelated benchmark.
---

# Reproduce paper results

Require an identified paper and a runnable audited model. Verify the primary paper,
supplement, authoritative source revision, and reported tables before spending
compute. Route an existing implementation uncertainty to `audit-model`; when the
model is absent, render the `paper-to-model` task instead of implementing it inside
an experiment-reproduction run.

## Map the protocol

Build a compact paper-to-repository map covering dataset name and version, split
boundaries, scaling, feature mode, lookback and horizons, covariates, loss, optimizer
and schedule, batch size, epochs and stopping, seeds, checkpoint selection, metric
formula and aggregation, baselines, and hardware-sensitive settings. Label every
field `aligned`, `adapted`, `unknown`, or `blocked`; do not silently fill paper gaps
with repository defaults.

Encode aligned settings in dedicated run configs with inheritance. Keep faithful
replication separate from controlled adaptations, and inspect the resolved matrix:

```bash
uv run tsf inspect --config <paper-run.toml>
```

Stop for a decision when missing data, licensing, ambiguous metrics, or infeasible
compute would materially change the claim.

## Execute and compare

Launch only when authorized, using `run-experiment` for concurrency and resource
handling. Preserve raw outputs, resolved configs, environment facts, seeds, and
failed runs. Aggregate only compatible cells with `analyze-results`.

Compare each local value with the matching paper cell using the same metric
direction and aggregation. Report paper value, local value, absolute and relative
difference, run count, uncertainty when available, and all protocol deviations.
Missing or failed cells remain visible.

Success means the experiment is rerunnable and the comparison is traceable; it
does not require matching the paper. Describe the outcome as reproduced only when
the recorded protocol and results support that claim, otherwise report a partial
replication or a blocked attempt.
