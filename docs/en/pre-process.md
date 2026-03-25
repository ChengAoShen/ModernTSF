# Pre-process datasets

`tool/pre_process.py` converts raw CSV files into pre-windowed `.npz` files for use with the `pre_processed` dataset. Pre-processing eliminates repeated windowing overhead during training and is useful for large datasets or many repeated runs.

## Output format

Each `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `x` | `(N, seq_len, C)` | Input windows |
| `y` | `(N, label_len + pred_len, C)` | Decoder target windows |
| `x_mark` | `(N, seq_len, 6)` | Input timestamps (year/month/day/weekday/hour/minute) |
| `y_mark` | `(N, label_len + pred_len, 6)` | Decoder timestamps |
| `scaler_mean` | `(C,)` | StandardScaler mean — only when `--scale` |
| `scaler_scale` | `(C,)` | StandardScaler scale — only when `--scale` |

Output files: `train.npz`, `val.npz`, `test.npz` in the specified `--output-dir`.

If no `date` column is present, `x_mark` and `y_mark` are zero-filled.

---

## Mode A: single CSV (auto-split)

Use when you have one CSV file. The script splits it into train/val/test automatically.

```bash
uv run python tool/pre_process.py \
    --input-csv dataset/ETT-small/ETTh1.csv \
    --output-dir dataset/ETTh1_npy \
    --seq-len 512 --label-len 0 --pred-len 96 \
    --features M --target OT --scale
```

Split ratios are controlled by `--split-ratio` (default `0.7,0.1,0.2`). The StandardScaler is fitted on the train split when `--scale` is set.

---

## Mode B: pre-split folder

Use when your data is already split into separate files.

```bash
uv run python tool/pre_process.py \
    --input-dir dataset/my_dataset \
    --output-dir dataset/my_dataset_npy \
    --seq-len 512 --label-len 0 --pred-len 96 \
    --features M --target OT --scale
```

The folder must contain `train.csv`, `val.csv`, and `test.csv` with the same column layout.

---

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input-csv PATH` | Mode A | — | Single CSV file (auto-split) |
| `--input-dir DIR` | Mode B | — | Folder with train/val/test CSVs |
| `--output-dir DIR` | Yes | — | Output directory for `.npz` files |
| `--seq-len INT` | Yes | — | Input sequence length |
| `--label-len INT` | Yes | — | Decoder label length |
| `--pred-len INT` | Yes | — | Prediction horizon |
| `--features` | No | `M` | `M`, `S`, or `MS` |
| `--target STR` | No | `OT` | Target column for `S`/`MS` mode |
| `--scale` / `--no-scale` | No | `--scale` | Apply StandardScaler |
| `--split-ratio T,V,TE` | No | `0.7,0.1,0.2` | Split ratios for Mode A |

`--input-csv` and `--input-dir` are mutually exclusive; exactly one is required.

---

## Using the output with ModernTSF

Set `dataset.name = "pre_processed"` and point `root_path` to the output directory:

```toml
[dataset]
name = "pre_processed"
root_path = "./dataset/ETTh1_npy"
data_path = ""

[dataset.params]
# No params required — all windowing was done by pre_process.py
```

See `configs/datasets/pre_processed.toml` for a ready-made template.

> **Note**: If you used `--scale` during pre-processing, the `.npz` files already contain the scaler. The `pre_processed` dataset loads `scaler_mean` / `scaler_scale` automatically and makes `inverse_transform` available. Set `task.inverse = true` in the run config to inverse-transform predictions before computing metrics.
