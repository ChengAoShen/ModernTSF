---
name: plot
description: Plot a bubble chart from an already-aggregated results CSV. Use when the user wants to visualize model performance as a bubble chart and already has a CSV file (e.g. produced by the `aggregate` skill or `tool/aggregate_results.py`).
---

## When to use

Invoke when the user has an aggregated CSV (typically `work_dirs/<dataset>/results_all.csv`) and wants a bubble chart. If the CSV does not yet exist, run the `aggregate` skill first.

Ask the user for:
- `CSV` — path to the aggregated results CSV (required)
- `X` — column name for the x-axis (required, e.g. `mse`)
- `Y` — column name for the y-axis (required, e.g. `mae`)
- `SIZE` — column name for bubble size (required, e.g. `total_params`)

Optional, ask only if relevant:
- `--size-scale` — `linear` (default), `sqrt`, or `log`
- `--x-scale` / `--y-scale` — `linear` (default) or `log`
- `--color-by` — column to group/color bubbles (default: `model`)
- `--label-by` — column to annotate points (default: `model`)
- `--no-labels` — flag to suppress point labels
- `--legend` — flag to show a legend
- `--title` — optional plot title string
- `--output` — output path (default: `work_dirs/plots/bubble_<csv-stem>.svg`)
- `--show` — flag to open an interactive plot window

## Command

```bash
uv run python tool/plot_bubble.py \
    --csv <CSV> \
    --x <X> \
    --y <Y> \
    --size <SIZE> \
    [--size-scale linear|sqrt|log] \
    [--x-scale linear|log] \
    [--y-scale linear|log] \
    [--color-by <COLUMN>] \
    [--label-by <COLUMN>] \
    [--no-labels] \
    [--legend] \
    [--title "<TITLE>"] \
    [--output <OUTPUT_PATH>] \
    [--show]
```

### Typical example

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/ETTh1/results_all.csv \
    --x mse \
    --y mae \
    --size total_params
```

Output is saved to `work_dirs/plots/bubble_<csv-stem>.svg` by default.

## Notes

- The CSV must contain the columns given to `--x`, `--y`, `--size`, and `--color-by` / `--label-by`.
- Non-numeric values in those columns are coerced; rows that remain non-numeric are dropped.
- `--size-scale log` and `--x-scale log` / `--y-scale log` silently drop rows with non-positive values.
- Bubble sizes are normalized to the range 30–300 pt² after applying the chosen scale.
- `tsf aggregate-plot` runs aggregation and plotting in one step: `uv run python tool/tsf.py aggregate-plot --dataset <name> --pred-len <len>`.

## Reference

See `docs/en/plot-bubble.md` for full parameter documentation.
