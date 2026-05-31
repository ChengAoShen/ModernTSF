# Bubble chart

Draw a bubble chart from a CSV file by selecting fields for x, y, and size. Colors are assigned per model by default. Typical input is `results_all.csv` produced by `aggregate_results.py`.

## Usage

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/<dataset>/results_all.csv \
    --x <x-field> \
    --y <y-field> \
    --size <size-field>
```

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--csv` | yes | — | Input CSV path |
| `--x` | yes | — | Field for x axis |
| `--y` | yes | — | Field for y axis |
| `--size` | yes | — | Field for bubble size |
| `--size-scale` | no | `linear` | Scale applied to size values before normalisation: `linear`, `sqrt`, or `log` |
| `--x-scale` | no | `linear` | Axis scale for x: `linear` or `log` |
| `--y-scale` | no | `linear` | Axis scale for y: `linear` or `log` |
| `--color-by` | no | `model` | Field used to assign bubble colors |
| `--label-by` | no | `model` | Field used to annotate each bubble |
| `--no-labels` | no | off | Disable per-bubble text labels |
| `--legend` | no | off | Show a legend |
| `--output` | no | `work_dirs/plots/bubble_<csv>.png` | Output image path |
| `--show` | no | off | Open an interactive plot window |
| `--title` | no | auto | Plot title; default is `<x> vs <y> \| size: <size> (<size-scale>)` |

## Scale options

### `--size-scale`

Controls how raw size values are transformed before being linearly mapped to the `[30, 300]` point-area range.

| Value | Effect | When to use |
|---|---|---|
| `linear` | No transformation; raw values used directly | Size field spans a small, uniform range |
| `sqrt` | `value ^ 0.5`; compresses large values, spreads small ones | Parameter counts or similar quantities that span one order of magnitude |
| `log` | `log10(value)`; extreme compression of large values | Quantities spanning multiple orders of magnitude (e.g. `total_params` from 10 K to 100 M) |

Rows with non-positive values are dropped when `log` is selected; rows with negative values are dropped when `sqrt` is selected.

### `--x-scale` / `--y-scale`

Sets the matplotlib axis scale. `log` uses a base-10 logarithmic axis. Rows with non-positive values on a `log` axis are removed before plotting.

## Common field choices

The fields below come from the CSV produced by `aggregate_results.py` (performance + profile merge).

| Field | Source | Typical use |
|---|---|---|
| `mse` | `performance.csv` | Accuracy — x or y axis |
| `mae` | `performance.csv` | Accuracy — x or y axis |
| `pred_len` | `performance.csv` | Grouping axis |
| `latency_avg_ms` | `profile.csv` | Inference cost — x axis |
| `throughput_samples_sec` | `profile.csv` | Throughput — x axis |
| `total_params` | `profile.csv` | Model size — size or axis |
| `peak_vram_mb` | `profile.csv` | Memory cost — size or axis |

## Examples

### Accuracy vs inference cost, size = model parameters

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/ETTh1/results_all.csv \
    --x latency_avg_ms \
    --y mse \
    --size total_params \
    --size-scale log \
    --x-scale log \
    --title "ETTh1: accuracy vs latency"
```

Both axes use `log` scale because latency and MSE can span an order of magnitude. `--size-scale log` prevents a single large model from dominating the visual area.

### MSE vs MAE, size = VRAM, save to custom path

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/weather/results_all.csv \
    --x mse \
    --y mae \
    --size peak_vram_mb \
    --size-scale sqrt \
    --legend \
    --output work_dirs/weather/mse_mae_vram.svg
```

`--size-scale sqrt` is a middle ground: it reduces visual dominance of memory-hungry models without the aggressiveness of `log`.

### Group by prediction horizon instead of model

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/ETTh1/results_all.csv \
    --x total_params \
    --y mse \
    --size latency_avg_ms \
    --color-by pred_len \
    --label-by model \
    --legend
```

## Notes

- Numeric values are auto-extracted from strings (e.g. `"2.5 ms"` → `2.5`).
- By default, labels are drawn near each bubble and the legend is hidden.
- Up to 20 distinct categories use the `tab20` colormap; 21–60 use `turbo`; more than 60 use `hsv`.
- The output directory is created automatically if it does not exist.
- For `log` scale (axis or size), rows with non-positive values are silently removed.
