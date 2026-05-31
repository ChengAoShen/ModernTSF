---
name: aggregate
description: Aggregate experiment results from work_dirs into a combined CSV and optionally plot a bubble chart. Use when the user wants to collect, summarize, filter, or visualize benchmark results for a dataset.
---

## When to use / what to ask

Ask the user:
1. Which dataset to aggregate (e.g. `ETTh1`, `electricity`)
2. Any filters to apply — optional (e.g. `pred_len=96`, `model~Linear`, `mse<=0.5`)
3. Whether to also generate a bubble chart (and if so, which axes)

## Step 1 — Aggregate results

```bash
uv run python tool/aggregate_results.py \
  --dataset <dataset> \
  --output work_dirs/<dataset>/results_all.csv
```

With filters:
```bash
uv run python tool/aggregate_results.py \
  --dataset <dataset> \
  --filter "pred_len=96,model~Linear" \
  --output work_dirs/<dataset>/results_filtered.csv
```

Optional flags:
- `--work-dir <path>` — root work directory (default: `./work_dirs`)
- `--perf-fields <fields>` — comma-separated fields from `performance.csv` (default: `model,seq_len,pred_len,mse,mae`)
- `--prof-fields <fields>` — comma-separated fields from `profile.csv` (default: `latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb`)

Filter operators: `=`, `!=`, `<`, `>`, `<=`, `>=`, `~` (substring match).

New metric columns are available in `performance.csv` and can be passed to
`--perf-fields` / `--metric-cols`: `mse`, `mae`, `corr`, `rse`, `wape`, `smape`, `mase`.

### TFB fairness collapse (optional)

Collapse per-seed runs into one row per `(model, pred_len)` and drop models that
are missing on too many cells, for a fair leaderboard:

```bash
uv run python tool/aggregate_results.py \
  --dataset <dataset> --collapse --aggregate mean \
  --metric-cols mse,mae --null-threshold 0.3
```

- `--collapse` — one row per `(model, pred_len)` instead of raw per-run rows (off by default).
- `--aggregate mean|median|max` — how `--collapse` combines a metric across seeds (default `mean`).
- `--null-threshold F` — exclude any model NaN/missing on more than fraction `F` of cells; dropped models are logged. Typical `0.3`. Unset disables exclusion.
- `--metric-cols <cols>` — metric columns the fairness policy aggregates/null-checks (default `mse,mae`).

## Step 2 — Plot bubble chart (optional)

```bash
uv run python tool/plot_bubble.py \
  --csv work_dirs/<dataset>/results_all.csv \
  --x <x_field> \
  --y <y_field> \
  --size <size_field>
```

Common axis choices: `mse`, `mae`, `latency_avg_ms`, `total_params`, `peak_vram_mb`.

Optional flags:
- `--size-scale linear|sqrt|log` (default: `linear`)
- `--x-scale linear|log` (default: `linear`)
- `--y-scale linear|log` (default: `linear`)
- `--color-by <field>` (default: `model`)
- `--label-by <field>` (default: `model`)
- `--no-labels` — disable point annotations
- `--legend` — show legend
- `--title <title>` — custom plot title
- `--output <path>` — output image path (default: `work_dirs/plots/bubble_<csv>.svg`)
- `--show` — open an interactive plot window

## One-shot path (aggregate + plot together)

```bash
[DATASET=<dataset>] [PRED_LEN=<pred_len>] [X=<x>] [Y=<y>] [SIZE=<size>] \
  [OUT_CSV=<csv_path>] [OUT_SVG=<svg_path>] \
  bash scripts/aggregate_and_plot.sh [<dataset>] [<pred_len>]
```

Positional args default to `ETTh1` and `96`; all options are also overridable via same-name env vars.

## Notes

- `aggregate_results.py` scans `work_dirs/<dataset>/*/performance.csv` and `profile.csv`.
- Profile data requires `evaluation.enable_profile = true` in the run config.
- Output CSV defaults to `work_dirs/<dataset>/results_all.csv` when `--output` is omitted.

See also: [docs/en/aggregate-results.md](../../docs/en/aggregate-results.md) and [docs/en/plot-bubble.md](../../docs/en/plot-bubble.md).
