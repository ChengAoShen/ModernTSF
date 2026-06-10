---
name: characteristics
description: Extract TFB-style statistical characteristics (trend strength, seasonality strength, stationarity) from a dataset. Use when the user wants to profile, describe, or quantify the trend/seasonality/stationarity of a time-series dataset before benchmarking.
---

## When to use

Profile a dataset's statistical properties — how trended, how seasonal, how stationary — before choosing models, or to report dataset-level stats. Works with any dataset config the project supports (single-file, custom, presplit, pre-processed, traffic bundles).

## Command

```bash
uv run python tool/dataset_characteristics.py \
    --config configs/datasets/etth1.toml --split train --per-channel
```

Outputs `trend_strength` and `seasonality_strength` (STL-style, 0–1) plus `stationarity` (ADF p-value via statsmodels) to `work_dirs/<dataset>/characteristics_<split>.csv`. `--period N` overrides the FFT-estimated seasonal period; omit `--per-channel` for dataset-level rows only.

## Reference

Full flags and metric definitions: `docs/en/dataset-characteristics.md`.
