---
name: rank
description: Rank models per pred_len/seed for a dataset, producing MSE and MAE leaderboard CSVs. Use when the user wants leaderboards, rankings, or to compare model performance across prediction horizons.
---

## When to use

Invoke after experiments have been run and `performance.csv` files exist under `work_dirs/`. Ask the user for the **dataset name** if not provided.

## Command

```bash
uv run python tool/rank_models.py \
    --dataset <DATASET> \
    [--input-root <work_dirs>] \
    [--out-mse <path/model_rankings_mse.csv>] \
    [--out-mae <path/model_rankings_mae.csv>] \
    [--out-long <path/model_rankings_long.csv>] \
    [--null-threshold 0.3] [--aggregate mean|median|max] [--fill-nan-with-mean]
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `ETTh1` | Dataset name to rank (must match folder name under `work_dirs/`) |
| `--input-root` | `work_dirs` | Root dir containing `<dataset>/<model>/performance.csv` files |
| `--out-mse` | `work_dirs/<DATASET>/model_rankings_mse.csv` | Wide MSE rankings CSV (model names by rank) |
| `--out-mae` | `work_dirs/<DATASET>/model_rankings_mae.csv` | Wide MAE rankings CSV (model names by rank) |
| `--out-long` | `work_dirs/<DATASET>/model_rankings_long.csv` | Long-form rankings CSV (suitable for downstream plotting) |

### TFB fairness flags (optional)

- `--null-threshold F` — exclude any model NaN/missing on more than fraction `F` of the `(pred_len, seed)` cells; dropped models are logged. Typical `0.3`. Unset disables exclusion.
- `--aggregate mean|median|max` — collapse duplicate metric values within a `(model, pred_len, seed)` cell (default `mean`; no-op without duplicates).
- `--fill-nan-with-mean` — after exclusion, fill remaining NaN metric cells with the metric's column mean before ranking (off by default).

## Notes

- Reads all `performance.csv` files found recursively under `--input-root`; each file must have columns `model`, `pred_len`, `seed`, `mse`, `mae`.
- Rankings are computed per `(pred_len, seed)` group; rank 1 = lowest metric value.
- Output CSVs are written to `work_dirs/<DATASET>/` by default; parent directories are created automatically.
- The long-form CSV (`--out-long`) contains `dataset`, `model`, `pred_len`, `seed`, `metric`, `value`, `rank` columns — useful for custom plotting.

## Docs

See `docs/en/rank-models.md` for full reference.
