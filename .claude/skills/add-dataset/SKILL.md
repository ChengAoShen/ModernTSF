---
name: add-dataset
description: Guide the user through adding a new dataset to the ModernTSF project. Use when the user wants to register a new dataset, integrate a CSV file or pre-split folder, or wire up a custom data source for benchmarking.
---

## When to use / what to ask

First ask the user:

1. **What is the dataset name?** (used for the config key and file names)
2. **Which pattern fits the data?**
   - **Pattern C (custom CSV) — preferred for a plain CSV.** One CSV (`date` column + numeric channels), split automatically by the built-in `custom` loader — config only, no code.
   - **Pattern B (pre-split)** — `train.csv`/`val.csv`/`test.csv` already in one folder — config only (`name = "presplit"`).
   - **Pattern A (single-file)** — unusual layout or synthetic generation needing a bespoke `_read_data` — new source files.
   - **Traffic / spatiotemporal bundle** — value array + adjacency matrix (PEMS, METR-LA) — convert with `tool/convert_traffic.py`, then a `cauair_st` config.

## Scaffold (one command)

```bash
uv run python tool/tsf.py new-dataset --name my_csv --pattern custom \
    --root-path ./dataset/my_csv --data-path my_csv.csv --target OT
# --pattern presplit  → config only (folder must hold train/val/test.csv)
# --pattern single    → also generates the loader + schema + DATASET_NAME_MAP entry
```

Put the data under `--root-path` and reference the new config from a run config via `extends`. For `single`, fill the real loader into `src/data/datasets/<name>.py::_read_data` (use `self._get_borders` for the split).

For a traffic bundle, build the node bundle first, then point a `cauair_st` config at it:

```bash
uv run python tool/convert_traffic.py \
    --values dataset/metr_la/metr_la.npz --values-key data \
    --adj dataset/metr_la/adj_mx.pkl --output-dir dataset/metr_la \
    --seq-len 12 --pred-len 12 --add-time --freq-min 5 --splits 0.7,0.1,0.2
```

## Verify

```bash
uv run modern-tsf --config <run config extending the new dataset>
uv run python tool/tsf.py aggregate-plot --dataset <name>
```

## Notes

- Pattern A requires a `date` column; Pattern B treats it as optional (zero timestamps if absent). The scaler is always fitted on train.
- Single-target mode (`features = "S"`) selects the channel via `target`.
- Graph configs can set `[dataset.params] adj_norm` to normalize the adjacency for graph models.
- After setup, offer to create a run config that uses the dataset.

## Reference

Complete TOML templates and per-file code for every pattern: `docs/en/add-dataset.md`; traffic bundles: `docs/en/datasets-traffic.md`.
