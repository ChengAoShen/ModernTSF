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

## Run a single config

```bash
uv run modern-tsf --config <config_path>
```

Example:

```bash
uv run modern-tsf --config configs/runs/run_single_data.toml
```

## Run multiple configs (shell script)

```bash
uv run python tool/tsf.py run [config ...] [--jobs N] [--gpus <ids>]
```

- `GPU_IDS` defaults to `0`; positional `config` args default to `configs/runs/run_single_data.toml`
- Pass `-h` for usage

## Preview a sweep before running

```bash
uv run python tool/inspect_config.py --config <config_path>
```

## After the run — aggregate results

```bash
uv run python tool/aggregate_results.py --dataset <dataset_name>
```

Or use the combined aggregate-and-plot script:

```bash
uv run python tool/tsf.py aggregate-plot --dataset <name> --pred-len <len>
```

## Optional config knobs

Set these inside the run config (not CLI flags):

- **Training tricks** — `[training.tricks]` supports `grad_clip`, `grad_accum`,
  `curriculum`, and aux-loss options for tougher training setups.
- **Rolling evaluation** — `[evaluation] strategy = "rolling"` switches from the
  default single-shot eval to RollingForecast over the test set.
- **Profiling** — `[evaluation] enable_profile = true` records params/MACs/latency.
- For ablation/hyperparameter sweeps and forecast case plots, use the `experiments` skill.

## Notes

- The only CLI flag for `modern-tsf` is `--config <path>` (required).
- Results are written to `work_dirs/<dataset>/<model>/performance.csv`.
- Offer to aggregate (and optionally plot a bubble chart) after the run completes.

## See also

`docs/en/aggregate-results.md`
