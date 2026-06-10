---
name: pre-process
description: Pre-process a dataset into pre-windowed .npz files for use with the ModernTSF `pre_processed` dataset type. Use when the user wants to convert CSV data to .npz format before training, or to set up a new dataset with pre-computed windows.
---

## When to use / what to ask

Converts a CSV into pre-windowed `train/val/test.npz` for the `pre_processed` dataset type. Ask for:

1. **Input** — a single CSV (`--input-csv`, auto-split `0.7,0.1,0.2` by default) or a folder already holding `train/val/test.csv` (`--input-dir`)
2. **Output dir** and **window sizes** (`--seq-len` / `--label-len` / `--pred-len`)
3. **Feature mode** (`M`|`S`|`MS`, default `M`; `S`/`MS` also need `--target`, default `OT`)

## Command

```bash
uv run python tool/pre_process.py \
    --input-csv dataset/ETT-small/ETTh1.csv \
    --output-dir dataset/ETTh1_npy \
    --seq-len 512 --label-len 0 --pred-len 96 --features M
```

Then point a dataset config at the output:

```toml
[dataset]
name = "pre_processed"
root_path = "<output-dir>"
data_path = ""
```

## Notes

- Window sizes baked into the `.npz` must match the values used at training time.
- Scaler is always fitted on train only; with scaling on (default), set `task.inverse = true` in the run config to inverse-transform predictions.

## Reference

Full flags (`--split-ratio`, `--no-scale`, output file contents): `docs/en/pre-process.md`.
