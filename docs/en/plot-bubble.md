# Bubble chart

Draw a bubble chart from a CSV file by selecting fields for x, y, and size. Colors are assigned per model by default.

## Usage

```bash
uv run python tool/plot_bubble.py --csv work_dirs/periodic/results_all.csv --x mse --y mae --size total_params
```

## Examples

```bash
uv run python tool/plot_bubble.py --csv work_dirs/periodic/results_all.csv --x latency_avg_ms --y mse --size total_params --size-scale log --x-scale log
```

## Arguments

- `--csv`: input CSV path.
- `--x`: field for x axis.
- `--y`: field for y axis.
- `--size`: field for bubble size.
- `--size-scale`: `linear`, `sqrt`, or `log` (default: `linear`).
- `--x-scale`: `linear` or `log` (default: `linear`).
- `--y-scale`: `linear` or `log` (default: `linear`).
- `--color-by`: field used to color groups (default: `model`).
- `--label-by`: field used to annotate points (default: `model`).
- `--no-labels`: disable point labels.
- `--legend`: show legend.
- `--output`: output image path (default: `work_dirs/plots/bubble_<csv>.svg`).
- `--show`: show the plot window.
- `--title`: optional plot title.

## Notes

- Numeric values are auto-extracted from strings (e.g., `"2.5 ms"`).
- By default, labels are drawn near each bubble and the legend is hidden.
- For `log` scaling, rows with non-positive values are removed.
