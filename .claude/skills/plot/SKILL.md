---
name: plot
description: Plot a bubble chart from an already-aggregated results CSV. Use when the user wants to visualize model performance as a bubble chart and already has a CSV file (e.g. produced by the `aggregate` skill or `tool/aggregate_results.py`).
---

## When to use / what to ask

The user has an aggregated CSV (typically `work_dirs/<dataset>/results_all.csv`) and wants a bubble chart. If the CSV does not exist yet, run the `aggregate` skill first.

Ask for the CSV path and the three axis columns: `--x` / `--y` / `--size` (e.g. `mse` / `mae` / `total_params`).

## Command

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/ETTh1/results_all.csv \
    --x mse --y mae --size total_params
```

Output defaults to `work_dirs/plots/bubble_<csv-stem>.svg`. Scales (`--x-scale`/`--y-scale`/`--size-scale`), grouping (`--color-by`/`--label-by`), `--title`, and `--output` are tunable — see `--help`.

`tsf aggregate-plot` runs aggregation + plotting in one step:
`uv run python tool/tsf.py aggregate-plot --dataset <name> --pred-len <len>`.

## Reference

Full flags and behavior (log scales silently drop non-positive rows; bubble sizes normalized to 30–300 pt²): `docs/en/plot-bubble.md`.
