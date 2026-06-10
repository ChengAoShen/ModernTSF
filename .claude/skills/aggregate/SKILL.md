---
name: aggregate
description: Aggregate experiment results from work_dirs into a combined CSV and optionally plot a bubble chart. Use when the user wants to collect, summarize, filter, or visualize benchmark results for a dataset.
---

## When to use / what to ask

Ask the user:
1. Which dataset to aggregate (e.g. `ETTh1`, `electricity`)
2. Optional filters (e.g. `pred_len=96`, `model~Linear`, `mse<=0.5` — operators `=`, `!=`, `<`, `>`, `<=`, `>=`, `~` substring)
3. Whether to also plot a bubble chart

## Commands

```bash
# Aggregate (default output: work_dirs/<dataset>/results_all.csv)
uv run python tool/aggregate_results.py --dataset <dataset> [--filter "pred_len=96,model~Linear"]

# One-shot aggregate + bubble chart
uv run python tool/tsf.py aggregate-plot --dataset <name> --pred-len <len>
```

For a fair leaderboard (TFB-style), add `--collapse --aggregate mean --null-threshold 0.3` — collapses per-seed runs into one row per `(model, pred_len)` and excludes models missing on too many cells (dropped models are logged).

## Notes

- Scans `work_dirs/<dataset>/*/performance.csv` and `profile.csv`; profile columns (latency, params, VRAM) require `evaluation.enable_profile = true` in the run config.
- Column selection (`--perf-fields`, `--prof-fields`, `--metric-cols`) supports all recorded metrics: `mse`, `mae`, `corr`, `rse`, `wape`, `smape`, `mase`.

## Reference

Full flags: `docs/en/aggregate-results.md`; chart options: `docs/en/plot-bubble.md` (or the `plot` skill).
