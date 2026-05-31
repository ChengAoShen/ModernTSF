---
name: characteristics
description: Extract TFB-style statistical characteristics (trend strength, seasonality strength, stationarity) from a dataset. Use when the user wants to profile, describe, or quantify the trend/seasonality/stationarity of a time-series dataset before benchmarking.
---

## When to use

Profile a dataset's statistical properties — how trended, how seasonal, how
stationary it is — to understand it before choosing models or to report
dataset-level stats. Wraps `tool/dataset_characteristics.py`.

## Command

```bash
uv run python tool/dataset_characteristics.py \
    --config <dataset_or_run_config.toml> \
    --split <train|val|test> \
    --out work_dirs/<dataset>/characteristics_<split>.csv
```

Example:

```bash
uv run python tool/dataset_characteristics.py \
    --config configs/datasets/etth1.toml --split train --per-channel
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--config` | (required) | Dataset TOML or full run TOML |
| `--split` | `train` | Split to analyse: `train`, `val`, or `test` |
| `--period N` | FFT-estimated | Seasonal period; if unset, taken from the dominant FFT frequency |
| `--per-channel` | off | Also emit one row per channel (otherwise dataset-level only) |
| `--out PATH` | `work_dirs/<dataset>/characteristics_<split>.csv` | Output CSV |

## Output columns

- `trend_strength` — STL-style `1 - Var(resid) / Var(resid + trend)` (moving-average trend).
- `seasonality_strength` — STL-style `1 - Var(resid) / Var(resid + seasonal)` (period-averaged seasonal).
- `stationarity` — ADF p-value if `statsmodels` is installed, else a moment-based proxy in `[0, 1]` (1.0 = perfectly stationary).

## Notes

- Works with any dataset config the project supports (single-file, custom, presplit, pre-processed, traffic bundles).
- Install `statsmodels` (already in deps) for the proper ADF stationarity test.
