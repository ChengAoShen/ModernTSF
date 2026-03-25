Visualize dataset samples from a TOML config in the ModernTSF project.

Ask the user:
1. Which dataset config to use (e.g. `configs/datasets/etth1.toml`)
2. Which split: `train`, `val`, or `test`
3. How many samples or a specific index
4. Which channels (optional — default: all)

## Basic usage

```bash
uv run python tool/visual_data.py \
  --config configs/datasets/etth1.toml \
  --split train \
  --num-samples 3 \
  --save work_dirs/plots/etth1_train.png
```

## Single sample by index

```bash
uv run python tool/visual_data.py \
  --config configs/datasets/etth1.toml \
  --split train \
  --index 0 \
  --channels 0,1,2 \
  --save work_dirs/plots/etth1_sample0.png
```

## Key arguments

- `--config`: path to a dataset TOML config (can be a full run config or dataset-only)
- `--split`: `train`, `val`, or `test`
- `--num-samples N`: plot N random samples (ignored if `--index` is given)
- `--index I`: plot a specific sample index
- `--channels`: comma-separated channel indices or `all` (default: `all`)
- `--save PATH`: output image path (default: `work_dirs/plots/<dataset>_<split>.png`)
- `--show`: open a display window
- `--seed`: random seed for sample selection

## Available dataset configs

`configs/datasets/`: `etth1`, `etth2`, `ettm1`, `ettm2`, `electricity`, `weather`, `traffic`, `solar`, `pre_processed`

## Notes

- The plot shows input series (solid line) with the forecast window starting after the dashed line.
- If the config only has `[dataset]`, task defaults are loaded from `configs/base.toml`.
