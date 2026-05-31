---
name: visualize
description: Visualize dataset samples from a TOML config in the ModernTSF project. Use when the user wants to plot, inspect, or preview time-series samples from a dataset split.
---

## When to use / what to ask

This skill plots **raw dataset samples** (input + forecast window) from a config.
For **forecast-vs-truth case plots from a trained model** (prediction studies),
use the `experiments` skill instead — it wraps `tool/visualize_predictions.py`.

Ask the user for:
1. Which dataset config to use (e.g. `configs/datasets/etth1.toml`) — any dataset-only or full run TOML works
2. Which split: `train`, `val`, or `test` (default: `train`)
3. How many samples (`--num-samples`) or a specific index (`--index`) — default: 3 samples
4. Which channels to plot (default: `all`)

## Commands

**Basic — plot N samples from a split:**
```bash
uv run python tool/visual_data.py \
  --config <config_path> \
  --split <train|val|test> \
  --num-samples <N> \
  --save work_dirs/plots/<dataset>_<split>.png
```

**Single sample by index with channel selection:**
```bash
uv run python tool/visual_data.py \
  --config <config_path> \
  --split <train|val|test> \
  --index <I> \
  --channels <0,1,2|all> \
  --save work_dirs/plots/<dataset>_sample<I>.png
```

**Random sampling with a seed:**
```bash
uv run python tool/visual_data.py \
  --config <config_path> \
  --split <train|val|test> \
  --num-samples <N> \
  --seed <S> \
  --save work_dirs/plots/<dataset>_<split>.png
```

## All flags

| Flag | Default | Description |
|---|---|---|
| `--config` | (required) | Path to a dataset TOML or full run TOML |
| `--split` | `train` | Split to load: `train`, `val`, or `test` |
| `--num-samples N` | `3` | Number of samples to plot (ignored when `--index` is set) |
| `--index I` | — | Plot a single specific sample index |
| `--channels` | `all` | Comma-separated channel indices or `all` |
| `--save PATH` | `work_dirs/plots/<dataset>_<split>.png` | Output image path |
| `--show` | off | Open an interactive display window |
| `--seed S` | — | Random seed for sample selection |

## Available dataset configs

`configs/datasets/`: `etth1`, `etth2`, `ettm1`, `ettm2`, `electricity`, `weather`, `traffic`, `solar`, `pre_processed`

## Notes

- The plot shows input series (solid line) and forecast window (after the dashed vertical line).
- If the config has only a `[dataset]` section (no `[task]`), task defaults are loaded from `configs/base.toml`.
- Output is saved to `work_dirs/plots/` by default; the directory is created automatically.

Docs: `docs/en/visualize-data.md`
