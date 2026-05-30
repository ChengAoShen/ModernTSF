---
name: gift-eval
description: Download GIFT-EVAL datasets from HuggingFace and run the full 53-dataset GIFT-EVAL benchmark sweep. Use when the user wants to run, reproduce, or evaluate against the GIFT-EVAL benchmark.
---

## When to use / what to ask

Ask the user:

1. **Where to store the data?** Default is `~/.cache/gift_eval`. A custom path saves re-downloading if the data already exists elsewhere.
2. **All 53 datasets or a subset?** Omit `--datasets` for all; pass specific names (e.g. `electricity/H m4_monthly`) to download fewer.
3. **Data already downloaded?** Use `--link-only` to skip the download and just create the symlink.

## Step 1 — Download datasets and create symlink

```bash
# All 53 datasets to default location:
uv run python tool/gift_eval_download.py

# Custom download location:
uv run python tool/gift_eval_download.py --output-dir /data/gift_eval

# Specific datasets only:
uv run python tool/gift_eval_download.py --datasets electricity/15T ett1/H m4_monthly

# Already downloaded — symlink only:
uv run python tool/gift_eval_download.py --link-only --output-dir /data/gift_eval

# List all available dataset names:
uv run python tool/gift_eval_download.py --list
```

This creates `./dataset/gift_eval -> <output-dir>` so that TOML configs referencing `root_path = "./dataset/gift_eval"` resolve automatically.

## Step 2 — Run the benchmark sweep

```bash
uv run modern-tsf --config configs/runs/gift_eval_sweep.toml
```

Or via the multi-config shell script (supports `GPU_IDS` env override):

```bash
[GPU_IDS=<ids>] bash scripts/run_multi_configs.sh configs/runs/gift_eval_sweep.toml
```

## Step 3 — Aggregate and plot results

```bash
# Aggregate for a specific GIFT-EVAL dataset:
uv run python tool/aggregate_results.py --dataset <gift_eval_dataset_name>

# Combined aggregate + bubble chart:
[DATASET=<name> PRED_LEN=<len>] bash scripts/aggregate_and_plot.sh [DATASET] [PRED_LEN]
```

## Notes

- `--output-dir` defaults to `~/.cache/gift_eval`; `--datasets` accepts the `base/freq` form shown by `--list`
- Unknown names passed to `--datasets` print a warning but do not abort
- The symlink at `dataset/gift_eval` is reused across runs; re-running with a different `--output-dir` updates it
- Results land in `work_dirs/<dataset>/<model>/performance.csv`
- Preview the sweep before running: `uv run python tool/inspect_config.py --config configs/runs/gift_eval_sweep.toml`

## See also

`docs/en/aggregate-results.md`
