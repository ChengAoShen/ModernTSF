# Dataset characteristics

ModernTSF ships a standalone, TFB-inspired analysis layer for quantifying the
properties of a dataset. It is purely diagnostic: it never touches training and
adds no new dependencies (numpy/scipy only).

`tool/dataset_characteristics.py`

## Basic usage

```bash
uv run python tool/dataset_characteristics.py --config configs/datasets/etth1.toml --split train
```

This loads the dataset split the same way `tool/visual_data.py` does (reusing
the dataset registry), computes characteristics, prints a table, and writes a
CSV. Per-channel rows are added with `--per-channel`.

## Key arguments

- `--config`: path to a TOML config. Can be a full run config or a dataset-only config.
- `--split`: `train`, `val`, or `test` (default `train`).
- `--period`: seasonal period. If unset, estimated from the dominant FFT frequency.
- `--per-channel`: also emit one row per channel (otherwise dataset-level only).
- `--out`: output CSV path. Defaults to `work_dirs/<dataset>/characteristics_<split>.csv`.

## Computed characteristics

| Column | Meaning |
|---|---|
| `period` | Seasonal period used for the STL-style decomposition (FFT-estimated or `--period`). |
| `trend_strength` | STL-style `1 - Var(resid) / Var(resid + trend)` using a moving-average trend. Range `[0, 1]`. |
| `seasonality_strength` | STL-style `1 - Var(resid) / Var(resid + seasonal)` using a period-averaged seasonal component. Range `[0, 1]`. |
| `stationarity` | Higher = more stationary. ADF `1 - p-value` if `statsmodels` is installed, otherwise a lightweight rolling mean/variance stability ratio. The `stationarity_method` column records which was used. |
| `shifting` | Absolute mean shift between the first and second half, normalised by the series std. |
| `transition` | Lag-1 autocorrelation. |
| `correlation` | Mean absolute pairwise channel correlation (dataset row only; `n/a` per channel). |

The dataset-level row aggregates the per-channel univariate statistics (mean
across channels), except `correlation`, which is computed across all channels.

## Stationarity note

`statsmodels` is **not** a project dependency. When it is absent (the default),
the `stationarity` column falls back to a lightweight rolling-moment stability
score and `stationarity_method` reads `rolling(n/a-statsmodels)`. If
`statsmodels` happens to be installed, the ADF test is used automatically.

## Example

```bash
uv run python tool/dataset_characteristics.py \
    --config configs/datasets/etth1.toml \
    --split train --per-channel --period 24 \
    --out work_dirs/ETTh1/characteristics_train.csv
```
