---
name: run-autoresearch
description: Run a bounded iterative forecasting research loop from a falsifiable hypothesis through experiments and evidence-backed conclusions. Use when autonomous experiment iteration is authorized; not for a single predefined run or paper implementation.
---

# Run bounded autoresearch

Require a research question, approved datasets, primary metric, hard run and time
budgets, output location, and authorization to execute experiments. Use
`design-experiment` to define a falsifiable comparison, fair baseline, seeds,
controls, and stopping criteria before spending compute.

Preview every matrix with `uv run tsf inspect --config <run.toml>`. Start with the
cheapest experiment that can reject the hypothesis, then use `run-experiment`.
Preserve resolved configs, seeds, environments, raw outputs, and failures; never
overwrite a costly run. Route failures through `diagnose-experiment` and compatible
results through `analyze-results`.

Change one experimental factor per iteration unless an interaction is the stated
question. Continue only when the previous evidence justifies the next run. Stop on
budget exhaustion, repeated infrastructure failure, invalid comparison, no
measurable progress, or a conclusion strong enough for the declared acceptance
criterion.

Return the hypothesis, run ledger, compatible metrics with uncertainty, failed or
excluded cells, limitations, and a stop/continue recommendation. Repository model
code, external publication, and additional task dispatch remain out of scope unless
separately authorized.
