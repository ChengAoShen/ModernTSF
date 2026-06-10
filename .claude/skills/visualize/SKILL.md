---
name: visualize
description: Visualize dataset samples from a TOML config in the ModernTSF project. Use when the user wants to plot, inspect, or preview time-series samples from a dataset split.
---

## When to use / what to ask

Plots **raw dataset samples** (input + forecast window) from a config. For **forecast-vs-truth case plots from a trained model**, use the `experiments` skill instead.

Ask for:
1. Which dataset config (e.g. `configs/datasets/etth1.toml`) — any dataset-only or full run TOML works
2. Which split: `train`, `val`, or `test` (default `train`)
3. How many samples (`--num-samples`, default 3) or a specific `--index`

## Command

```bash
uv run python tool/visual_data.py \
  --config configs/datasets/etth1.toml --split train --num-samples 3
```

Saves to `work_dirs/plots/<dataset>_<split>.png` by default (input series solid, forecast window after the dashed line). Other knobs: `--channels 0,1,2` (default `all`), `--seed S`, `--save PATH`, `--show`.

## Reference

Full flags: `docs/en/visualize-data.md`. Dataset configs live in `configs/datasets/`.
