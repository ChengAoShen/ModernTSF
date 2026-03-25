# Visualize datasets

ModernTSF includes a standalone script for quick dataset inspection:

`tool/visual_data.py`

## Basic usage

```bash
uv run python tool/visual_data.py --config configs/datasets/etth1.toml --split train --num-samples 3 --save work_dirs/plots/etth1.png
```

## Key arguments

- `--config`: path to a TOML config. Can be a full run config or dataset-only config.
- `--split`: `train`, `val`, or `test`.
- `--num-samples`: number of samples to plot if `--index` is not provided.
- `--index`: plot a specific sample index.
- `--channels`: comma-separated channel indices, or `all`.
- `--save`: output image path. Defaults to `work_dirs/plots/<dataset>_<split>.png`.
- `--show`: open a window to display the figure.
- `--seed`: random seed for sample selection.

## Single-channel example

```bash
uv run python tool/visual_data.py --config configs/datasets/etth1.toml --split train --channels 0 --save work_dirs/plots/etth1_ch0.png
```

## Dataset-only configs

If the config only contains `[dataset]`, the script uses task defaults from `configs/base.toml`. You can override by adding a `[task]` section to the same config.
