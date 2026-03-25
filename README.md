# ModernTSF — Modern Time Series Forecasting

A structured, engineering-grade time-series forecasting benchmark. AI-friendly, docs-first, and easy to extend — run complex experiments fast with TOML composition, profiling, and rich visualization. Built for reproducible sweeps, clean experiment management, and quick iteration from idea to results.

## Highlights

- **TOML-first configs**: compose datasets, models, and sweeps for complex experiments with clear, versionable configs
- **31 models out of the box**: from simple linear baselines to modern Transformers, MLPs, and more
- **Fast to run**: single configs, model sweeps, dataset sweeps, multi-axis sweeps, and explicit `sweep.extend` order
- **Profiling-ready and visualization-friendly**: aggregate results, track metrics, and plot charts quickly
- **AI-friendly and readable**: clear docs and code structure that make VibeCode workflows fast and low-friction
- **Extensible by design**: plug in new datasets, models, and metrics with minimal wiring

## Quick start

Create the environment and install dependencies:

```bash
uv sync --python 3.12
```

Run a single dataset experiment:

```bash
uv run modern-tsf --config configs/runs/run_single_data.toml
```

Run model sweep, dataset sweep, or multi-axis sweep:

```bash
uv run modern-tsf --config configs/runs/sweep_model.toml
uv run modern-tsf --config configs/runs/sweep_data.toml
uv run modern-tsf --config configs/runs/multi_sweep.toml
```

`sweep.extend` expands first, then the remaining `[sweep]` keys. Total runs are the product of all extend axes and sweep values.

Aggregate results and plot a bubble chart:

```bash
uv run python tool/aggregate_results.py --dataset ETTh1
uv run python tool/plot_bubble.py --csv work_dirs/ETTh1/results_all.csv --x mse --y mae --size total_params
```

Rank models per pred_len/seed:

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

## Available Models (31)

| Name | Category |
|---|---|
| `Linear`, `DLinear`, `NLinear`, `RLinear` | Linear baselines |
| `CrossLinear`, `MixLinear` | Linear variants |
| `Autoformer`, `FEDformer`, `PatchTST`, `iTransformer` | Transformer-based |
| `PatchMLP`, `xPatch`, `TSMixer`, `LightTS` | MLP / Patch-based |
| `TimesNet` | CNN (2D time-frequency) |
| `TimeMixer` | Multi-scale mixing |
| `SegRNN` | RNN (segmented) |
| `FITS`, `SparseTSF`, `CycleNet`, `TiDE`, `SCINet` | Modern forecasters |
| `Amplifier`, `TimeBase`, `TimeBridge`, `TimeEmb` | Architecture variants |
| `PaiFilter`, `TexFilter` | Filter-based |
| `SVTime`, `CMoS`, `PWS` | Other |

All models are available as TOML configs in `configs/models/`. Model params are defined in `src/models/<name>/schema.py`.

## Available Datasets

| Config | Description |
|---|---|
| `configs/datasets/etth1.toml` | ETT hourly 1 |
| `configs/datasets/etth2.toml` | ETT hourly 2 |
| `configs/datasets/ettm1.toml` | ETT minute 1 |
| `configs/datasets/ettm2.toml` | ETT minute 2 |
| `configs/datasets/electricity.toml` | Electricity consumption (321 channels) |
| `configs/datasets/weather.toml` | Weather multivariate (21 channels) |
| `configs/datasets/traffic.toml` | Road traffic (862 channels) |
| `configs/datasets/solar.toml` | Solar power (text file) |
| `configs/datasets/pre_processed.toml` | Pre-windowed `.npz` files |

Pre-split and synthetic (`periodic`, `trend`) datasets are also supported — see `docs/en/add-dataset.md`.

## Tools

| Script | Purpose |
|---|---|
| `tool/inspect_config.py` | Preview config expansion (sweep counts, datasets, models) |
| `tool/aggregate_results.py` | Merge performance + profile CSVs for a dataset |
| `tool/plot_bubble.py` | Draw bubble chart from aggregated CSV |
| `tool/rank_models.py` | Rank models per pred_len / seed |
| `tool/visual_data.py` | Visualise dataset samples from a TOML config |
| `tool/pre_process.py` | Convert CSVs to pre-windowed `.npz` files |
| `tool/tsne_ecg.py` | t-SNE comparison for ECG datasets |

Shell scripts for common workflows are in `scripts/`.

## Documentation (English)

- Parameters reference: `docs/en/params.md`
- Config loading and usage: `docs/en/configs.md`
- Add a new model: `docs/en/add-model.md`
- Add a new dataset: `docs/en/add-dataset.md`
- Pre-process datasets: `docs/en/pre-process.md`
- Models reference: `docs/en/models.md`
- Visualize datasets: `docs/en/visualize-data.md`
- Aggregate results: `docs/en/aggregate-results.md`
- Model rankings: `docs/en/rank-models.md`
- Bubble chart: `docs/en/plot-bubble.md`
