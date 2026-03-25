# 数据集预处理

`tool/pre_process.py` 将原始 CSV 文件转换为预窗口化的 `.npz` 文件，供 `pre_processed` 数据集使用。预处理可以消除训练时重复窗口化的开销，适用于大数据集或需要多次重复实验的场景。

## 输出格式

每个 `.npz` 文件包含：

| 键 | 形状 | 说明 |
|---|---|---|
| `x` | `(N, seq_len, C)` | 输入窗口 |
| `y` | `(N, label_len + pred_len, C)` | 解码器目标窗口 |
| `x_mark` | `(N, seq_len, 6)` | 输入时间戳（年/月/日/星期/时/分） |
| `y_mark` | `(N, label_len + pred_len, 6)` | 解码器时间戳 |
| `scaler_mean` | `(C,)` | StandardScaler 均值 — 仅当 `--scale` 时存在 |
| `scaler_scale` | `(C,)` | StandardScaler 标准差 — 仅当 `--scale` 时存在 |

输出文件：`train.npz`、`val.npz`、`test.npz`，写入 `--output-dir` 指定目录。

若数据无 `date` 列，`x_mark` 和 `y_mark` 将以零填充。

---

## 模式 A：单 CSV 文件（自动切分）

适用于拥有单个 CSV 文件的情况，脚本会自动切分。

```bash
uv run python tool/pre_process.py \
    --input-csv dataset/ETT-small/ETTh1.csv \
    --output-dir dataset/ETTh1_npy \
    --seq-len 512 --label-len 0 --pred-len 96 \
    --features M --target OT --scale
```

切分比例由 `--split-ratio` 控制（默认 `0.7,0.1,0.2`）。开启 `--scale` 时，StandardScaler 仅在训练集上拟合。

---

## 模式 B：预切分文件夹

适用于数据已提前切分为独立文件的情况。

```bash
uv run python tool/pre_process.py \
    --input-dir dataset/my_dataset \
    --output-dir dataset/my_dataset_npy \
    --seq-len 512 --label-len 0 --pred-len 96 \
    --features M --target OT --scale
```

文件夹须包含 `train.csv`、`val.csv`、`test.csv`，且列结构相同。

---

## 参数说明

| 参数 | 是否必填 | 默认值 | 说明 |
|---|---|---|---|
| `--input-csv PATH` | 模式 A | — | 单个 CSV 文件（自动切分） |
| `--input-dir DIR` | 模式 B | — | 含 train/val/test CSV 的文件夹 |
| `--output-dir DIR` | 是 | — | `.npz` 文件输出目录 |
| `--seq-len INT` | 是 | — | 输入序列长度 |
| `--label-len INT` | 是 | — | 解码器 label 长度 |
| `--pred-len INT` | 是 | — | 预测长度 |
| `--features` | 否 | `M` | `M`、`S` 或 `MS` |
| `--target STR` | 否 | `OT` | `S`/`MS` 模式的目标列名 |
| `--scale` / `--no-scale` | 否 | `--scale` | 是否应用 StandardScaler |
| `--split-ratio T,V,TE` | 否 | `0.7,0.1,0.2` | 模式 A 的切分比例 |

`--input-csv` 与 `--input-dir` 互斥，必须提供其中一个。

---

## 在 ModernTSF 中使用

将 `dataset.name` 设为 `"pre_processed"`，并将 `root_path` 指向输出目录：

```toml
[dataset]
name = "pre_processed"
root_path = "./dataset/ETTh1_npy"
data_path = ""

[dataset.params]
# 无需额外参数 — 窗口化已由 pre_process.py 完成
```

参考现成模板：`configs/datasets/pre_processed.toml`。

> **注意**：如果预处理时使用了 `--scale`，`.npz` 文件中已包含 scaler 参数，`pre_processed` 数据集会自动加载并提供 `inverse_transform`。在 run config 中设置 `task.inverse = true` 可在计算指标前对预测结果反归一化。
