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
[GPU_IDS=<ids>] bash scripts/run_multi_configs.sh [config ...]
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
[DATASET=<name> PRED_LEN=<len>] bash scripts/aggregate_and_plot.sh [DATASET] [PRED_LEN]
```

## Notes

- The only CLI flag for `modern-tsf` is `--config <path>` (required).
- Results are written to `work_dirs/<dataset>/<model>/performance.csv`.
- Offer to aggregate (and optionally plot a bubble chart) after the run completes.

## See also

`docs/en/aggregate-results.md`
