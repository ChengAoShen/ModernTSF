# Model rankings

Compute model rankings from `performance.csv` for each `pred_len` and `seed`, and export wide tables where each setting is a column and each cell is a model name.

## Usage

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

## Arguments

- `--dataset`: dataset name (default: `ETTh1`).
- `--input-root`: root work directory (default: `./work_dirs`).
- `--out-mse`: MSE ranking table output (default: `work_dirs/<dataset>/model_rankings_mse.csv`).
- `--out-mae`: MAE ranking table output (default: `work_dirs/<dataset>/model_rankings_mae.csv`).
- `--out-long`: long table output (default: `work_dirs/<dataset>/model_rankings_long.csv`).

## Outputs

- Wide table: columns named `pl<pred_len>_seed<seed>`, rows are ranks (`1` is best).
- Long table: each row includes `model, pred_len, seed, metric, value, rank` for plotting or filtering.
