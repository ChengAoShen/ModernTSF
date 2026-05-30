---
name: sweep
description: Batch-run one or more ModernTSF experiment configs sequentially via scripts/run_multi_configs.sh. Use when the user wants to launch several TOML run configs in one go, train across multiple sweeps, or run experiments on a specific GPU.
---

## When to use / what to ask

Use this skill when the user wants to run multiple experiment configs in sequence (or even a single config via the helper script). Before running, confirm:

1. **Config paths** — one or more TOML files under `configs/runs/`. Defaults to `configs/runs/run_single_data.toml` when none are given.
2. **GPU** — which GPU id(s) to use. Defaults to `0`. Pass a comma-separated list for multi-GPU visibility (e.g. `0,1`).

## Command

```bash
# Single config (default GPU 0)
bash scripts/run_multi_configs.sh configs/runs/run_single_data.toml

# Multiple configs on GPU 1
GPU_IDS=1 bash scripts/run_multi_configs.sh configs/runs/sweep_data.toml configs/runs/sweep_model.toml

# Multiple configs, multi-GPU visibility
GPU_IDS=0,1 bash scripts/run_multi_configs.sh configs/runs/sweep_data.toml configs/runs/sweep_model.toml
```

The script runs each config in order via:

```
CUDA_VISIBLE_DEVICES=<GPU_IDS> uv run modern-tsf --config <config>
```

## Notes

- The script uses `set -euo pipefail`; it aborts on the first failure.
- Paths are resolved relative to the repo root (`ROOT_DIR` is auto-detected).
- To preview what a sweep config expands to (datasets × models × pred_lens) before launching, run:
  ```bash
  uv run python tool/inspect_config.py --config <config>
  ```
- Results land in `work_dirs/<dataset>/<model>/performance.csv` after each run.

## Reference

See `docs/en/configs.md` for config structure, sweep syntax, and `extends` chains.
