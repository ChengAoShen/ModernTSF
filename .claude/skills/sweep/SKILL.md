---
name: sweep
description: Batch-run one or more ModernTSF experiment configs (concurrently) via `tool/tsf.py run`. Use when the user wants to launch several TOML run configs in one go, train across multiple sweeps, or run experiments on specific GPUs.
---

## When to use / what to ask

Use this skill when the user wants to run multiple experiment configs (or a single
config) through the unified runner. Before running, confirm:

1. **Config paths** — one or more TOML files under `configs/runs/`. Defaults to
   `configs/runs/run_single_data.toml` when none are given.
2. **Concurrency** — `--jobs N` runs configs in parallel (default `1`). Only raise
   it when the configs fit together (e.g. one per GPU); heavy trainings on one GPU
   should stay at `--jobs 1`.
3. **GPUs** — `--gpus 0,1` round-robins `CUDA_VISIBLE_DEVICES` across the jobs.

## Command

```bash
# Single config
uv run python tool/tsf.py run configs/runs/run_single_data.toml

# Two configs in parallel, one per GPU
uv run python tool/tsf.py run configs/runs/sweep_data.toml configs/runs/sweep_model.toml --jobs 2 --gpus 0,1

# Several configs, all on GPU 1, sequentially
uv run python tool/tsf.py run configs/runs/a.toml configs/runs/b.toml --gpus 1
```

`tsf run` reports `OK` / `FAIL` per config and exits non-zero if any config fails.

## Notes

- Each config runs as `CUDA_VISIBLE_DEVICES=<gpu> modern-tsf --config <config>`.
- To preview what a sweep config expands to (datasets × models × pred_lens) before
  launching, run:
  ```bash
  uv run python tool/tsf.py inspect --config <config>
  ```
- Results land in `work_dirs/<dataset>/<model>/performance.csv` after each run.

## Reference

See `docs/en/scripts.md` for the full `tsf` command reference and `docs/en/configs.md`
for config structure, sweep syntax, and `extends` chains.
