# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Commands

```bash
# Install / sync dependencies (Python 3.12, CUDA 12.4 build of PyTorch)
uv sync --python 3.12

# Run an experiment
uv run modern-tsf --config configs/runs/run_single_data.toml

# Preview config expansion (sweep counts, covered datasets/models)
uv run python tool/inspect_config.py --config configs/runs/multi_sweep.toml

# Aggregate performance + profile CSVs for a dataset
uv run python tool/aggregate_results.py --dataset ETTh1

# Plot a bubble chart from aggregated results
uv run python tool/plot_bubble.py --csv work_dirs/ETTh1/results_all.csv --x mse --y mae --size total_params

# Rank models per pred_len/seed
uv run python tool/rank_models.py --dataset ETTh1

# Visualise dataset samples
uv run python tool/visual_data.py --config configs/datasets/etth1.toml --split train --num-samples 3

# Pre-process a CSV into pre-windowed .npz files for the pre_processed dataset
uv run python tool/pre_process.py \
    --input-csv dataset/ETT-small/ETTh1.csv \
    --output-dir dataset/ETTh1_npy \
    --seq-len 512 --label-len 0 --pred-len 96 --features M
```

There are no automated tests and no linting config. All source packages live under `src/`, which is the package root (`package-dir = {"" = "src"}`), so `import benchmark`, `import data`, and `import models` all resolve from there.

## Available Datasets

| Config | Name key | Description |
|---|---|---|
| `configs/datasets/etth1.toml` | `ETTh1` | ETT hourly dataset 1 |
| `configs/datasets/etth2.toml` | `ETTh2` | ETT hourly dataset 2 |
| `configs/datasets/ettm1.toml` | `ETTm1` | ETT minute dataset 1 |
| `configs/datasets/ettm2.toml` | `ETTm2` | ETT minute dataset 2 |
| `configs/datasets/electricity.toml` | `electricity` | Electricity consumption |
| `configs/datasets/weather.toml` | `weather` | Weather multivariate |
| `configs/datasets/traffic.toml` | `traffic` | Road traffic |
| `configs/datasets/solar.toml` | `solar` | Solar power (text file) |
| `configs/datasets/pre_processed.toml` | `pre_processed` | Pre-windowed .npz files |

Synthetic datasets (`periodic`, `trend`) have source code under `src/data/datasets/` but no config file by default — create `configs/datasets/<name>.toml` as needed.

## Available Models (31)

| Config | Name key | Category |
|---|---|---|
| `Linear.toml` | `Linear` | Linear |
| `DLinear.toml` | `DLinear` | Linear (decomposition) |
| `NLinear.toml` | `NLinear` | Linear (normalised) |
| `RLinear.toml` | `RLinear` | Linear (RevIN) |
| `CrossLinear.toml` | `CrossLinear` | Linear (cross-channel) |
| `MixLinear.toml` | `MixLinear` | Linear (mixed) |
| `Autoformer.toml` | `Autoformer` | Transformer |
| `FEDformer.toml` | `FEDformer` | Transformer (frequency) |
| `PatchTST.toml` | `PatchTST` | Transformer (patch) |
| `iTransformer.toml` | `iTransformer` | Transformer (inverted) |
| `PatchMLP.toml` | `PatchMLP` | MLP (patch) |
| `xPatch.toml` | `xPatch` | MLP (patch, extended) |
| `TSMixer.toml` | `TSMixer` | MLP-Mixer |
| `LightTS.toml` | `LightTS` | Lightweight MLP |
| `TimesNet.toml` | `TimesNet` | CNN (2D time-freq) |
| `TimeMixer.toml` | `TimeMixer` | Multi-scale mixing |
| `SegRNN.toml` | `SegRNN` | RNN (segmented) |
| `FITS.toml` | `FITS` | Frequency interpolation |
| `SparseTSF.toml` | `SparseTSF` | Sparse forecaster |
| `CycleNet.toml` | `CycleNet` | Cycle-aware network |
| `TiDE.toml` | `TiDE` | Dense encoder-decoder |
| `SCINet.toml` | `SCINet` | Sample convolution |
| `Amplifier.toml` | `Amplifier` | Amplifier-based |
| `TimeBase.toml` | `TimeBase` | Time-based |
| `TimeBridge.toml` | `TimeBridge` | Bridge architecture |
| `TimeEmb.toml` | `TimeEmb` | Time-embedding enhanced |
| `PaiFilter.toml` | `PaiFilter` | Filter-based |
| `TexFilter.toml` | `TexFilter` | Filter-based |
| `SVTime.toml` | `SVTime` | Singular value |
| `CMoS.toml` | `CMoS` | Channel mixing |
| `PWS.toml` | `PWS` | Patch-wise |

## Architecture

### Config → Registry → Runner pipeline

The CLI entry point (`src/benchmark/cli.py`) drives everything in three steps:

1. **Load configs** — `benchmark.config.load_config(path)` reads a TOML file, resolves `extends` chains via deep-merge, expands `[sweep]` and `[sweep.extend]` into a cartesian product, and validates each expanded dict against `RootConfig` (Pydantic). Returns a list of `LoadedConfig` objects.

2. **Register components** — `register_from_config(config)` lazily imports and calls `register()` on the dataset, model, and metric modules referenced by name. Registration is idempotent (tracked in module-level sets). The maps that drive this are `DATASET_NAME_MAP`, `MODEL_NAME_MAP` in their respective registry modules.

3. **Run sweep** — `run_sweep(configs)` iterates the list and calls `run_one(config, raw, sweep_keys)` for each. `run_one` builds three DataLoaders (train/val/test), constructs the model, trains with early stopping, evaluates, and writes a CSV summary row to `work_dirs/<dataset>/<model>/performance.csv`.

### Registry pattern

Every extensible component (datasets, models, metrics, losses) uses the same pattern:
- A `*_REGISTRY` singleton in `src/benchmark/registry/`
- A `*_NAME_MAP` dict mapping string name → dotted module path
- Each module exposes a `register()` function that calls `REGISTRY.register(name, cls_or_factory, schema)`
- Schemas are Pydantic models that validate `dataset.params` / `model.params` from the TOML

### Data loading

`src/data/provider.py::build_data_loader` is the single factory used by `run_one`. It looks up the dataset class by name, passes `root_path`, `data_path`, `size=(seq_len, label_len, pred_len)`, `flag`, `features`, and unpacked `dataset_params` to the constructor.

Three dataset patterns exist:
- **Single-file** (`ForecastingDataset` subclass): inherits `_get_borders` for ratio-based splitting, implements `_read_data`. Examples: `Dataset_Custom`, `Dataset_ETT_hour`, `Dataset_Solar`.
- **Pre-split** (`Dataset_PreSplit`, direct `Dataset` subclass): reads `train.csv`/`val.csv`/`test.csv` from `root_path`. Use `name = "presplit"` and `data_path = ""` in config. Scaler is always fitted on `train.csv`.
- **Pre-processed** (`Dataset_PreProcessed`): reads pre-windowed `.npz` files produced by `tool/pre_process.py`. Use `name = "pre_processed"` and `data_path = ""`.

`__getitem__` always returns `(input_series, output_series, input_stamp, output_stamp)` as float32 numpy arrays. Timestamps are always real arrays (zero-filled when no `date` column), never `None`, to keep PyTorch's collate safe.

### Model interface

All models receive `(x, x_mark, dec_inp, dec_mark)` from `_call_model` in `trainer.py`. Models that don't use temporal marks should accept `*args` in `forward`. The factory lambda in each `registry.py` receives `(cfg: RootConfig, params: dict)`.

### Config inheritance

TOML files compose via `extends = [list of paths]` resolved relative to the file. Merge order: earlier files in the list are base, later files override. The current file overrides everything. `[sweep.extend]` named axes load entire TOML files as extra override layers, expanding via cartesian product before the regular `[sweep]` keys expand.

## Key files for extending the project

| Task | File(s) |
|---|---|
| Add a dataset (single-file) | `src/data/datasets/<name>.py`, `src/data/schemas/datasets/<name>.py`, `src/benchmark/registry/datasets.py` (`DATASET_NAME_MAP`), `configs/datasets/<name>.toml` |
| Add a dataset (pre-split) | `configs/datasets/<name>.toml` only — use `name = "presplit"` |
| Add a dataset (pre-processed .npz) | Run `tool/pre_process.py`, then `configs/datasets/<name>.toml` with `name = "pre_processed"` |
| Add a model | `src/models/<name>/model.py`, `schema.py`, `registry.py`, `src/benchmark/registry/models.py` (`MODEL_NAME_MAP`), `configs/models/<name>.toml` |
| Add a metric or loss | `src/benchmark/registry/metrics.py` or `losses.py` |
| Change training loop | `src/benchmark/runner/trainer.py` |
| Change evaluation | `src/benchmark/runner/evaluator.py` |
| Change config schema | `src/benchmark/config/schema/` |

## Scripts

Shell scripts for common workflows are in `scripts/`:

- `scripts/run_multi_configs.sh` — run one or more experiment configs sequentially with a specific GPU
- `scripts/aggregate_and_plot.sh` — aggregate results for a dataset then generate a bubble chart

Edit the `DATASET`, `PRED_LEN`, and `GPU_IDS` variables at the top of each script before running.

## Detailed docs

- `docs/en/` — English reference (params, configs, add-dataset, add-model, tools)
- `docs/zh-CN/` — Chinese mirror (same content, kept in sync)

Key doc files:
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
