---
name: run-autoresearch
description: Run a bounded iterative forecasting research loop from a falsifiable hypothesis through experiments and evidence-backed conclusions. Use when autonomous experiment iteration is authorized; not for a single predefined run or paper implementation.
---

# Run bounded autoresearch

Require a research question, approved datasets, primary metric, hard run and time
budgets, output location, and authorization to execute experiments. Use
`design-experiment` to define a falsifiable comparison, fair baseline, seeds,
controls, and stopping criteria before spending compute.

Start from the rendered `autoresearch` Harness when no round was supplied:

```bash
uv run tsf agent task start autoresearch --set 'question=<question>' --json
```

This command prepares the round and prompt; it does not dispatch an external
Agent. Continue in the current Agent or pass the rendered prompt to the chosen
Harness explicitly.

Use its round id for every experiment and record only useful hypotheses,
decisions, observations, failures, and conclusions. Do not create parallel memory
formats or copy raw results into notes; run artifacts and full logs already exist.

Preview every matrix with `uv run tsf inspect --config <run.toml>`. Start with the
cheapest experiment that can reject the hypothesis, then use `run-experiment`
with `--round <round-id>`.
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
