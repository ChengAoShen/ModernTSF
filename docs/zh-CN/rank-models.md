# 模型排名

根据 `performance.csv` 计算各模型在不同 `pred_len` 与 `seed` 下的排名，并输出按 setting 展开的宽表（每列一个 setting，单元格为模型名）。

## 用法

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

## 参数

- `--dataset`：要筛选的数据集名称（默认 `ETTh1`）。
- `--input-root`：包含 `<dataset>/<model>/performance.csv` 子目录的根路径（默认 `./work_dirs`）。
- `--out-mse`：宽表 MSE 排名输出路径（默认 `work_dirs/<dataset>/model_rankings_mse.csv`）。
- `--out-mae`：宽表 MAE 排名输出路径（默认 `work_dirs/<dataset>/model_rankings_mae.csv`）。
- `--out-long`：长表排名输出路径（默认 `work_dirs/<dataset>/model_rankings_long.csv`）。
- `--null-threshold`：TFB 公平性。排除在超过该比例的 `(pred_len, seed)` 单元上为 NaN/缺失的模型。默认未设置（不排除，保持原行为）。典型值：`0.3`。
- `--aggregate {mean,median,max}`：TFB 公平性。当同一 `(model, pred_len, seed)` 单元存在重复时如何折叠多个指标值（默认 `mean`）。无重复行时为空操作。
- `--fill-nan-with-mean`：TFB 公平性。在按 `--null-threshold` 排除模型后，用各指标在存活行上的列均值填充剩余 NaN 单元，再进行排名。默认关闭。

## 输入

工具会遍历 `<input-root>/**/performance.csv` 并拼接所有文件。每个 `performance.csv` 必须包含 `model`、`pred_len`、`seed`、`mse`、`mae` 列。若文件中不含 `dataset` 列，则从上级目录名推断（即 `work_dirs/<dataset>/<model>/performance.csv`）。

## 输出格式

### 宽表（`model_rankings_mse.csv`、`model_rankings_mae.csv`）

每个指标对应一张表。列名格式为 `pl<pred_len>_seed<seed>`（如 `pl96_seed0`），先按 `pred_len` 后按 `seed` 排序。行号即排名（第 1 行 = 排名第 1 = 最优）。每个单元格为该 setting 下达到该排名的模型名。

MSE 宽表示例：

| rank | pl96_seed0  | pl192_seed0 | pl96_seed1  |
|------|-------------|-------------|-------------|
| 1    | PatchTST    | TimeMixer   | PatchTST    |
| 2    | TimeMixer   | PatchTST    | DLinear     |
| 3    | DLinear     | DLinear     | TimeMixer   |

### 长表（`model_rankings_long.csv`）

每行对应一个 (model, pred_len, seed, metric) 组合。列包含：`dataset`、`model`、`pred_len`、`seed`、`metric`、`value`、`rank`。适合下游过滤、绘图或聚合排名统计。

长表行示例：

| dataset | model    | pred_len | seed | metric | value  | rank |
|---------|----------|----------|------|--------|--------|------|
| ETTh1   | PatchTST | 96       | 0    | mse    | 0.3821 | 1    |
| ETTh1   | DLinear  | 96       | 0    | mse    | 0.3974 | 2    |
| ETTh1   | PatchTST | 96       | 0    | mae    | 0.3952 | 2    |

## 示例

对 ETTh1 数据集使用默认输出路径进行排名：

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

自定义输出路径：

```bash
uv run python tool/rank_models.py \
    --dataset weather \
    --out-mse results/weather_mse_ranks.csv \
    --out-mae results/weather_mae_ranks.csv \
    --out-long results/weather_ranks_long.csv
```

从非默认工作目录读取：

```bash
uv run python tool/rank_models.py \
    --dataset ETTm1 \
    --input-root /mnt/experiments/work_dirs
```

## 注意事项

- 排名按 `(dataset, pred_len, seed, metric)` 分组计算。并列时采用最小排名（`method="min"`）。
- 默认从 `work_dirs` 读取数据；请先通过 `uv run modern-tsf --config ...` 运行实验以生成 `performance.csv`。
- 长表可直接作为 `plot_bubble.py` 的输入，也方便自定义 pandas 分析。
