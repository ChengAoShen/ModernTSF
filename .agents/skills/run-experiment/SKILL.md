---
name: run-experiment
description: Preview and run one or more ModernTSF experiment or sweep configurations. Use for training, evaluation, ablations, hyperparameter grids, concurrency, or GPU assignment; not for quick contract-only checks.
---

# Run experiments

The Agent owns the design and interpretation. Use the library for config
validation, execution, budgets and recovery; do not reimplement these guarantees
with ad hoc subprocesses or manually edited manifests. Call Python APIs when
available; CLI examples below are equivalent optional adapters.

Preview before launch:

```bash
uv run tsf run configs/runs/<run>.toml --dry-run --json
uv run tsf run configs/runs/<run>.toml [--round <round-id>]
```

Associate a run with a research round when one was supplied; do not create a
round for a one-off run unless the task asks for persistent research context.
Use `--jobs N` for independent configs and `--gpus 0,1` only after checking memory and device intent. Keep sweeps in TOML. Before long runs, verify data, output location, seeds, horizons, strategy, and profiling. Report successful/failed configs and `work_dirs/` artifacts; do not silently restart or overwrite costly runs.

Route failed, unstable, or suspect runs to `diagnose-experiment`. Route compatible,
complete outputs to `analyze-results`; execution itself does not establish a fair
comparison.

For GIFT-Eval, inspect `uv run tsf dataset gift-download --help`, obtain only the
requested data, and preview `configs/runs/gift_eval_sweep.toml` before launch.
Record dataset versions, horizons, model compatibility, compute budget, and the
missing-series policy; incomplete cells must remain visible during analysis.

For optional budgets, GPU queueing, tracking, cancellation, or interrupted-run
recovery, read [execution controls](references/execution.md). Do not load advanced
controls for an ordinary one-off experiment.
