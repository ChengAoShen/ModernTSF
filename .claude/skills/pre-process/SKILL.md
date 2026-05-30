---
name: pre-process
description: Pre-process a dataset into pre-windowed .npz files for use with the ModernTSF `pre_processed` dataset type. Use when the user wants to convert CSV data to .npz format before training, or to set up a new dataset with pre-computed windows.
---

## When to use / what to ask

Ask the user for:

1. **Input source** — one of:
   - A single CSV file (will be auto-split) → `--input-csv`
   - A folder already containing `train.csv`, `val.csv`, `test.csv` → `--input-dir`
2. **Output directory** (`--output-dir`)
3. **Window sizes**: `--seq-len`, `--label-len`, `--pred-len`
4. **Feature mode** (`M`, `S`, or `MS`; default `M`) and `--target` column if using `S`/`MS` (default `OT`)
5. **Scaling** — `--scale` (default on) or `--no-scale`
6. *(Mode A only)* Split ratio if non-default — `--split-ratio T,V,TE` (default `0.7,0.1,0.2`)

## Mode A — single CSV (auto-split)

```bash
uv run python tool/pre_process.py \
    --input-csv <path/to/data.csv> \
    --output-dir <path/to/output> \
    --seq-len <N> --label-len <N> --pred-len <N> \
    --features <M|S|MS> [--target <col>] \
    [--scale | --no-scale] \
    [--split-ratio <T,V,TE>]
```

Default split ratio: `0.7,0.1,0.2`.

## Mode B — pre-split folder

```bash
uv run python tool/pre_process.py \
    --input-dir <path/to/folder> \
    --output-dir <path/to/output> \
    --seq-len <N> --label-len <N> --pred-len <N> \
    --features <M|S|MS> [--target <col>] \
    [--scale | --no-scale]
```

Folder must contain `train.csv`, `val.csv`, `test.csv`.

## Output

Writes `train.npz`, `val.npz`, `test.npz` to `--output-dir`. Each file contains:
`x` (inputs), `y` (targets), `x_mark`, `y_mark`, and (when `--scale`) `scaler_mean`/`scaler_scale`.
Scaler is always fitted on `train` only.

## After pre-processing — create a dataset config

```toml
[dataset]
name = "pre_processed"
root_path = "<path/to/output>"
data_path = ""

[dataset.params]
# No extra params required
```

## Notes

- `seq_len`, `label_len`, and `pred_len` must match the values used at training time.
- If `--scale` is used, set `task.inverse = true` in the run config to inverse-transform predictions.
- `--features M` or `MS` uses all non-`date` columns; `--features S` uses only `--target`.

See `docs/en/pre-process.md` for the full argument reference.
