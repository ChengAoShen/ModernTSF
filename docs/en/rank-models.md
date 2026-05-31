# Model rankings

Compute model rankings from `performance.csv` for each `pred_len` and `seed`, and export wide tables where each setting is a column and each cell is a model name.

## Usage

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

## Arguments

- `--dataset`: dataset name to filter (default: `ETTh1`).
- `--input-root`: root directory containing `<dataset>/<model>/performance.csv` subfolders (default: `./work_dirs`).
- `--out-mse`: wide MSE ranking table output path (default: `work_dirs/<dataset>/model_rankings_mse.csv`).
- `--out-mae`: wide MAE ranking table output path (default: `work_dirs/<dataset>/model_rankings_mae.csv`).
- `--out-long`: long ranking table output path (default: `work_dirs/<dataset>/model_rankings_long.csv`).
- `--null-threshold`: TFB fairness. Exclude any model that is NaN/missing on more than this fraction of the `(pred_len, seed)` cells for the dataset. Unset (default) disables exclusion and preserves prior behavior. Typical value: `0.3`.
- `--aggregate {mean,median,max}`: TFB fairness. How to collapse multiple metric values within the same `(model, pred_len, seed)` cell when duplicates exist (default: `mean`). A no-op when there are no duplicate rows.
- `--fill-nan-with-mean`: TFB fairness. After excluding models over `--null-threshold`, fill any remaining NaN metric cells with that metric's column mean (per metric, over surviving rows) before ranking. Off by default.

## Input

The tool globs `<input-root>/**/performance.csv` and concatenates all files. Each `performance.csv` must contain the columns `model`, `pred_len`, `seed`, `mse`, and `mae`. If a file has no `dataset` column the dataset name is inferred from the grandparent directory name (i.e. `work_dirs/<dataset>/<model>/performance.csv`).

## Output formats

### Wide tables (`model_rankings_mse.csv`, `model_rankings_mae.csv`)

One table per metric. Columns are named `pl<pred_len>_seed<seed>` (e.g. `pl96_seed0`), sorted first by `pred_len` then `seed`. Rows are ranks (row 1 = rank 1 = best). Each cell contains the model name that achieved that rank for that setting.

Example MSE wide table:

| rank | pl96_seed0  | pl192_seed0 | pl96_seed1  |
|------|-------------|-------------|-------------|
| 1    | PatchTST    | TimeMixer   | PatchTST    |
| 2    | TimeMixer   | PatchTST    | DLinear     |
| 3    | DLinear     | DLinear     | TimeMixer   |

### Long table (`model_rankings_long.csv`)

Each row represents one (model, pred_len, seed, metric) combination. Columns: `dataset`, `model`, `pred_len`, `seed`, `metric`, `value`, `rank`. Useful for downstream filtering, plotting, or computing aggregate rank statistics.

Example long table rows:

| dataset | model    | pred_len | seed | metric | value  | rank |
|---------|----------|----------|------|--------|--------|------|
| ETTh1   | PatchTST | 96       | 0    | mse    | 0.3821 | 1    |
| ETTh1   | DLinear  | 96       | 0    | mse    | 0.3974 | 2    |
| ETTh1   | PatchTST | 96       | 0    | mae    | 0.3952 | 2    |

## Examples

Rank models for ETTh1 using default output paths:

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

Rank models and save outputs to custom paths:

```bash
uv run python tool/rank_models.py \
    --dataset weather \
    --out-mse results/weather_mse_ranks.csv \
    --out-mae results/weather_mae_ranks.csv \
    --out-long results/weather_ranks_long.csv
```

Read from a non-default work directory:

```bash
uv run python tool/rank_models.py \
    --dataset ETTm1 \
    --input-root /mnt/experiments/work_dirs
```

## Notes

- Rankings are computed per `(dataset, pred_len, seed, metric)` group. Ties receive the same rank (`method="min"`).
- The tool reads from `work_dirs` by default; run experiments first with `uv run modern-tsf --config ...` to populate `performance.csv` files.
- The long table is a convenient input for `plot_bubble.py` or custom pandas analysis.
