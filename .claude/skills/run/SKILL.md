---
name: run
description: Run a benchmark experiment using a TOML config file. Use when the user wants to run, train, or sweep experiments — single dataset/model or multi-axis sweeps — against any of the available configs/runs/ configs.
---

## When to use / what to ask

If the user has not specified a config, ask which experiment they want to run and suggest:

| Intent | Config |
|---|---|
| Single dataset + model | `configs/runs/run_single_data.toml` |
| Sweep over models | `configs/runs/sweep_model.toml` |
| Sweep over datasets | `configs/runs/sweep_data.toml` |
| Multi-axis sweep | `configs/runs/multi_sweep.toml` |
| GIFT-EVAL benchmark | `configs/runs/gift_eval_sweep.toml` |

## Commands

```bash
# Preview what a sweep expands to (run count, datasets, models)
uv run python tool/inspect_config.py --config <config_path>

# Run a single config (the only CLI flag is --config)
uv run modern-tsf --config <config_path>

# Run several configs / pick GPUs (see the `sweep` skill)
uv run python tool/tsf.py run <config ...> [--jobs N] [--gpus 0,1]
```

## Config knobs (set in the TOML, not the CLI)

- `[training.tricks]` — `grad_clip`, `grad_accum`, `curriculum`, aux-loss options.
- `[evaluation] strategy = "rolling"` — rolling forecast instead of single-shot eval.
- `[evaluation] enable_profile = true` — record params/MACs/latency per run.
- For ablation/hyperparameter sweeps and forecast case plots, use the `experiments` skill.

## Notes

- Results land in `work_dirs/<dataset>/<model>/performance.csv`; offer to aggregate (and optionally plot) afterwards via the `aggregate` skill.

## Reference

Config structure, `extends`, and sweep syntax: `docs/en/configs.md`; `tsf` runner: `docs/en/scripts.md`.
