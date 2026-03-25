# 模型排名

根据 `performance.csv` 计算各模型在不同 `pred_len` 与 `seed` 下的排名，并输出按 setting 展开的宽表（每列一个 setting，单元格为模型名）。

## 用法

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

## 参数

- `--dataset`：数据集名称（默认 `ETTh1`）。
- `--input-root`：工作目录根路径（默认 `./work_dirs`）。
- `--out-mse`：MSE 排名表输出路径（默认 `work_dirs/<dataset>/model_rankings_mse.csv`）。
- `--out-mae`：MAE 排名表输出路径（默认 `work_dirs/<dataset>/model_rankings_mae.csv`）。
- `--out-long`：长表输出路径（默认 `work_dirs/<dataset>/model_rankings_long.csv`）。

## 输出

- 宽表：列名为 `pl<pred_len>_seed<seed>`，行号为排名（`1` 为最好）。
- 长表：每行包含 `model, pred_len, seed, metric, value, rank` 等字段，便于可视化或筛选。
