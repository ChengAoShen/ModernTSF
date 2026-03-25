Pre-process a dataset into pre-windowed .npz files for use with the ModernTSF `pre_processed` dataset.

Ask the user:
1. Input: a single CSV file (`--input-csv`) or a folder with pre-split CSVs (`--input-dir`)
2. Output directory (`--output-dir`)
3. Window sizes: `seq_len`, `label_len`, `pred_len`
4. Feature mode (`M`, `S`, or `MS`) and target column if needed
5. Whether to apply scaling (default: yes)

## Mode A — single CSV (auto-split)

```bash
uv run python tool/pre_process.py \
    --input-csv <path/to/data.csv> \
    --output-dir <path/to/output> \
    --seq-len <N> --label-len <N> --pred-len <N> \
    --features M --scale
```

Default split: `0.7,0.1,0.2`. Override with `--split-ratio 0.6,0.2,0.2`.

## Mode B — pre-split folder (train/val/test CSVs)

```bash
uv run python tool/pre_process.py \
    --input-dir <path/to/folder> \
    --output-dir <path/to/output> \
    --seq-len <N> --label-len <N> --pred-len <N> \
    --features M --scale
```

Folder must contain `train.csv`, `val.csv`, `test.csv`.

## Output

Writes `train.npz`, `val.npz`, `test.npz` to `--output-dir`. Each file contains:
`x` (inputs), `y` (targets), `x_mark`, `y_mark`, and optionally `scaler_mean`/`scaler_scale`.

## After pre-processing — create a dataset config

```toml
[dataset]
name = "pre_processed"
root_path = "<path/to/output>"
data_path = ""

[dataset.params]
# No params required
```

## Notes

- `seq_len`, `label_len`, and `pred_len` must match what will be used at training time.
- If `--scale` is used, set `task.inverse = true` in the run config to inverse-transform predictions.
- See `docs/en/pre-process.md` for full argument reference.
