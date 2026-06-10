---
name: rank
description: Rank models per pred_len/seed for a dataset, producing MSE and MAE leaderboard CSVs. Use when the user wants leaderboards, rankings, or to compare model performance across prediction horizons.
---

## When to use

After experiments have produced `performance.csv` files under `work_dirs/`. Ask the user for the **dataset name** if not provided.

## Command

```bash
uv run python tool/rank_models.py --dataset <DATASET>
```

Writes `model_rankings_mse.csv`, `model_rankings_mae.csv`, and `model_rankings_long.csv` to `work_dirs/<DATASET>/`. Rankings are computed per `(pred_len, seed)` group; rank 1 = lowest metric value.

For a fair leaderboard (TFB-style), add `--null-threshold 0.3` (exclude models missing on too many cells), `--aggregate mean|median|max` (collapse duplicates), and/or `--fill-nan-with-mean`.

## Reference

Full flags, input requirements, and output formats: `docs/en/rank-models.md` (or `tool/rank_models.py --help`).
