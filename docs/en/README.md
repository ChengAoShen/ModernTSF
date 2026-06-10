# Documentation index

← [Back to repo root](../../README.md)

## Setup

| Doc | Description |
|-----|-------------|
| [setup-env.md](setup-env.md) | Detect the machine's GPU/CUDA and install the matching PyTorch backend with uv via `UV_TORCH_BACKEND`. |

## Reference

| Doc | Description |
|-----|-------------|
| [params.md](params.md) | All TOML fields and their meaning, keyed to `configs/base.toml` and the Pydantic schemas. |
| [configs.md](configs.md) | Config loading pipeline: `extends` inheritance, `[sweep]` expansion, and validation. |
| [models.md](models.md) | Catalogue of all 172 available models with architecture notes and key hyper-parameters. |
| [task-modes.md](task-modes.md) | Data settings selected by `task.mode`: `time_series`, `spatiotemporal`, and `covariate`. |

## How-to

| Doc | Description |
|-----|-------------|
| [add-model.md](add-model.md) | Step-by-step guide for adding a new model: package layout, schema, registry entry, and TOML config. |
| [add-dataset.md](add-dataset.md) | Step-by-step guide for adding a new dataset: single-file, pre-split, and pre-processed variants. |
| [datasets-traffic.md](datasets-traffic.md) | Convert and run METR-LA / PEMS-BAY / PEMS0x traffic graph bundles through the `cauair_st` node loader. |
| [pre-process.md](pre-process.md) | Pre-window a CSV into `.npz` files with `tool/pre_process.py` for use with the `pre_processed` dataset. |
| [experiments.md](experiments.md) | One-click experiments: launch sweeps, aggregate, rank, and plot forecast vs ground-truth case studies. |

## Tools

| Doc | Description |
|-----|-------------|
| [inspect-config.md](inspect-config.md) | Preview sweep expansion (run count, datasets, models) for a config with `tool/inspect_config.py`. |
| [aggregate-results.md](aggregate-results.md) | Merge `performance.csv` and `profile.csv` per dataset into a single CSV with `tool/aggregate_results.py`. |
| [plot-bubble.md](plot-bubble.md) | Generate a bubble chart from an aggregated CSV with `tool/plot_bubble.py`. |
| [rank-models.md](rank-models.md) | Rank models by metric per `pred_len`/seed with `tool/rank_models.py`. |
| [visualize-data.md](visualize-data.md) | Plot dataset samples from a TOML config with `tool/visual_data.py`. |
| [dataset-characteristics.md](dataset-characteristics.md) | Extract TFB-style dataset characteristics (trend/seasonality/stationarity/...) with `tool/dataset_characteristics.py`. |
| [gift-eval.md](gift-eval.md) | Download GIFT-EVAL datasets with `tool/gift_eval_download.py` and run the 53-dataset sweep. |
| [scripts.md](scripts.md) | Unified `tsf` tooling (scaffold / smoke / run / aggregate-plot) + `detect_hardware.sh`. |
