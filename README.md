<div align="center">

# 🚀 ModernTSF

**Modern Time Series Forecasting**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Models: 99](https://img.shields.io/badge/models-99-orange.svg)](#-available-models-99)
[![Datasets: 60+](https://img.shields.io/badge/datasets-60+-purple.svg)](#-available-datasets)
[![GIFT-EVAL](https://img.shields.io/badge/GIFT--EVAL-53_configs-blueviolet.svg)](#-gift-eval-benchmark)

A structured, engineering-grade time-series forecasting benchmark.
AI-friendly, docs-first, and easy to extend — run complex experiments fast
with TOML composition, profiling, and rich visualization.

[**English**](README.md) | [**中文**](README_zh.md)

</div>

---

## ✨ Highlights

- 📝 **TOML-first configs** — compose datasets, models, and sweeps for complex experiments with clear, versionable configs
- 🧠 **99 models out of the box** — from simple linear baselines to modern Transformers, MLPs, spatiotemporal and air-quality models
- 🎛️ **Three forecasting data settings** — `time_series`, `spatiotemporal`, and `covariate`, selectable per run
- 📊 **60+ datasets** — 9 classic benchmarks + 53 GIFT-EVAL configurations across 23 domains and 10 frequencies
- ⚡ **Fast to run** — single configs, model sweeps, dataset sweeps, multi-axis sweeps, and explicit `sweep.extend` order
- 📈 **Profiling & visualization** — aggregate results, track metrics, and plot charts quickly
- 🤖 **AI-friendly** — clear docs and code structure that make VibeCode workflows fast and low-friction
- 🔌 **Extensible by design** — plug in new datasets, models, and metrics with minimal wiring

---

## 🎛️ Task modes

All tasks are **forecasting**; `task.mode` picks the data setting. The default
is `time_series`, so existing configs are unchanged.

| Mode | Batch | Target | Example |
|---|---|---|---|
| `time_series` | `(B, T, C)` value | all channels | any CSV dataset |
| `spatiotemporal` | `(B, T, N, 1+F)` value + per-node covariates | value of `N` nodes | `synthetic_st`, `cauair_st` |
| `covariate` | spatiotemporal + **future** covariates | value of `N` nodes | `cauair_st` |

See `docs/en/task-modes.md` (or `docs/zh-CN/task-modes.md`) for details and
model/mode compatibility.

---

## 🏁 Quick Start

Create the environment and install dependencies. The PyTorch build (CPU or a
specific CUDA `cuXXX`) is chosen at install time — let uv auto-detect your GPU:

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12   # or cu124 / cu121 / cpu …
```

> New machine or GPU? Run `bash scripts/detect_hardware.sh` to detect your CUDA
> backend, or use the `setup-env` skill. See [setup-env.md](docs/en/setup-env.md).

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

> `sweep.extend` expands first, then the remaining `[sweep]` keys. Total runs are the product of all extend axes and sweep values.

Aggregate results and plot a bubble chart:

```bash
uv run python tool/aggregate_results.py --dataset ETTh1
uv run python tool/plot_bubble.py --csv work_dirs/ETTh1/results_all.csv --x mse --y mae --size total_params
```

Rank models per `pred_len` / seed:

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

---

## 🧠 Available Models (99)

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
| `MoFo`, `PHAT` | Periodic transformers (time series) |
| `BiST`, `MAGE`, `STOP` | Spatiotemporal |
| `CauAir`, `AirCade` | Air-quality (future covariates) |

All models are available as TOML configs in `configs/models/`. Model params are defined in `src/models/<name>/schema.py`.

The last seven models are ported from the [PoorOtterBob](https://github.com/PoorOtterBob)
repositories and run as standard univariate-channel forecasters here. Their
adapters convert ModernTSF's `(x_enc, x_mark_enc, x_dec, x_mark_dec)` batch into
each model's native layout — see `src/models/_external/marks.py`. The
spatiotemporal and air-quality models consume a `(B, T, N, 1+F)` tensor where the
value channel is augmented with `F = 2` normalized calendar features
(time-of-day, day-of-week); the air-quality models additionally take the future
calendar features as covariates. `PHAT`'s upstream repo omits its core
`PHAT_Attention` module, which is reconstructed from the paper
(ICLR 2026, arXiv:2602.00654) in `src/models/phat/layers/PHAT_Attention.py`.
AirCade trains with a frequency-domain MAE (`loss = "freq_mae"`); the rest
default to MAE. A tiny end-to-end smoke run for each lives in
`configs/runs/smoke_*.toml` (see `scripts/make_smoke_data.py`).

---

## 📊 Available Datasets

### Classic Benchmarks

| Config | Description |
|---|---|
| `configs/datasets/etth1.toml` | ETT hourly 1 |
| `configs/datasets/etth2.toml` | ETT hourly 2 |
| `configs/datasets/ettm1.toml` | ETT minute 1 |
| `configs/datasets/ettm2.toml` | ETT minute 2 |
| `configs/datasets/electricity.toml` | Electricity consumption (321 channels) |
| `configs/datasets/weather.toml` | Weather multivariate (21 channels) |
| `configs/datasets/traffic.toml` | Road traffic (862 channels) |
| `configs/datasets/solar.toml` | Solar power |
| `configs/datasets/pre_processed.toml` | Pre-windowed `.npz` files |

Pre-split and synthetic (`periodic`, `trend`) datasets are also supported — see `docs/en/add-dataset.md`.

### Spatiotemporal & Air-Quality

Node-structured datasets for the `spatiotemporal` and `covariate` task modes
(see [Task modes](#-task-modes)):

| Config | Description |
|---|---|
| `configs/datasets/synthetic_st.toml` | Synthetic node series with calendar covariates `[time_in_day, day_in_week]` |
| `configs/datasets/cauair_ccaq_st.toml` | CauAir / CCAQ air quality (209 nodes, meteorology covariates) — spatiotemporal layout |
| `configs/datasets/cauair_ccaq_ts.toml` | Same CauAir data as a plain forecasting dataset (nodes → channels) |

CauAir's `.npz` bundles (`his.npz`, `idx_{train,val,test}.npy`, `adj_mx.npy`)
are loaded by `cauair_st` / `cauair_ts`; place them under `dataset/<name>/`.

### 🏆 GIFT-EVAL Benchmark

ModernTSF natively supports the [**GIFT-EVAL**](https://huggingface.co/datasets/Salesforce/GiftEval) benchmark — **53 dataset configurations** spanning **23 base datasets**, **10 frequencies** (from secondly to monthly), and **7 domains** (energy, traffic, weather, finance, and more).

<details>
<summary><b>📋 Full GIFT-EVAL dataset list (click to expand)</b></summary>

| Dataset | Frequencies | Type |
|---|---|---|
| electricity | 15T, D, H, W | Univariate |
| ett1, ett2 | 15T, D, H, W | Multivariate (7-dim) |
| solar | 10T, D, H, W | Univariate |
| LOOP_SEATTLE | 5T, D, H | Univariate |
| jena_weather | 10T | Multivariate (21-dim) |
| M_DENSE | D, H | Univariate |
| SZ_TAXI | 15T, H | Univariate |
| bitbrains_fast_storage | 5T, H | Multivariate (2-dim) |
| bitbrains_rnd | 5T, H | Multivariate (2-dim) |
| bizitobs_application | 10S | Multivariate (2-dim) |
| bizitobs_l2c | 5T, H | Multivariate (7-dim) |
| bizitobs_service | 10S | Multivariate (2-dim) |
| hierarchical_sales | D, W | Univariate |
| kdd_cup_2018_with_missing | D, H | Univariate |
| saugeenday | D, M, W | Univariate |
| us_births | D, M, W | Univariate |
| m4_daily, m4_hourly, m4_monthly | — | Univariate |
| m4_quarterly, m4_weekly, m4_yearly | — | Univariate |
| car_parts_with_missing | M | Univariate |
| covid_deaths | D | Univariate |
| hospital | M | Univariate |
| restaurant | D | Univariate |
| temperature_rain_with_missing | D | Univariate |

</details>

**Quick setup:**

```bash
# Download all GIFT-EVAL datasets (choose your own location)
uv run python tool/gift_eval_download.py --output-dir /your/path

# Or link existing data
uv run python tool/gift_eval_download.py --link-only --output-dir /path/to/GiftEval

# Run full GIFT-EVAL sweep (short term, all 53 datasets)
uv run modern-tsf --config configs/runs/gift_eval_sweep.toml
```

Each dataset TOML uses GIFT-EVAL **short-term** prediction lengths by default. Medium (10x) and long (15x) terms are noted in each config file — just update `pred_len` to switch.

---

## 🛠️ Tools

| Script | Purpose |
|---|---|
| `tool/inspect_config.py` | Preview config expansion (sweep counts, datasets, models) |
| `tool/aggregate_results.py` | Merge performance + profile CSVs for a dataset |
| `tool/plot_bubble.py` | Draw bubble chart from aggregated CSV |
| `tool/rank_models.py` | Rank models per pred_len / seed |
| `tool/visual_data.py` | Visualise dataset samples from a TOML config |
| `tool/visualize_predictions.py` | Plot forecast vs ground-truth case studies for a trained run |
| `tool/dataset_characteristics.py` | Extract TFB-style dataset characteristics (trend / seasonality / stationarity) |
| `tool/convert_traffic.py` | Build a traffic / spatiotemporal node bundle (values + adjacency) for `cauair_st` |
| `tool/pre_process.py` | Convert CSVs to pre-windowed `.npz` files |
| `tool/gift_eval_download.py` | Download GIFT-EVAL datasets + create symlink |

### Workflow scripts

Parameterized shell wrappers for common workflows live in `scripts/` — positional args with env-var overrides (`-h`/`--help` prints usage):

```bash
# Run one or more configs sequentially (default: configs/runs/run_single_data.toml)
[GPU_IDS=<ids>] bash scripts/run_multi_configs.sh [config ...]

# Aggregate a dataset/pred_len then plot a bubble chart
[DATASET=… PRED_LEN=… X=… Y=… SIZE=… OUT_CSV=… OUT_SVG=…] \
  bash scripts/aggregate_and_plot.sh [DATASET] [PRED_LEN]
```

`run_multi_configs.sh` takes any number of TOML config paths (env `GPU_IDS`, default `0`). `aggregate_and_plot.sh` takes positional `DATASET` (default `ETTh1`) and `PRED_LEN` (default `96`), each overridable by the same-name env var, plus `X`/`Y`/`SIZE` and `OUT_CSV`/`OUT_SVG` env overrides. `detect_hardware.sh` reports your GPU/CUDA and recommends a `UV_TORCH_BACKEND` tag. See [scripts.md](docs/en/scripts.md).

### 🤖 Agent Skills

This repo ships [Claude Code](https://claude.ai/code) skills under `.claude/skills/` — `setup-env`, `run`, `experiments`, `aggregate`, `visualize`, `characteristics`, `pre-process`, `add-dataset`, `add-model`, `inspect`, `rank`, `plot`, `gift-eval`, and `sweep` — that wrap these tools for agent or human use via `/<name>`.

---

## 📖 Documentation

- 🇬🇧 [English docs](docs/en/README.md) — parameters, configs, add-model, add-dataset, tools
- 🇨🇳 [中文文档](docs/zh-CN/README.md) — 参数、配置、添加模型、添加数据集、工具

| Topic | English | 中文 |
|---|---|---|
| Environment setup (GPU/CUDA) | [setup-env.md](docs/en/setup-env.md) | [setup-env.md](docs/zh-CN/setup-env.md) |
| Parameters reference | [params.md](docs/en/params.md) | [params.md](docs/zh-CN/params.md) |
| Config loading | [configs.md](docs/en/configs.md) | [configs.md](docs/zh-CN/configs.md) |
| One-click experiments | [experiments.md](docs/en/experiments.md) | [experiments.md](docs/zh-CN/experiments.md) |
| Inspect config | [inspect-config.md](docs/en/inspect-config.md) | [inspect-config.md](docs/zh-CN/inspect-config.md) |
| Task modes | [task-modes.md](docs/en/task-modes.md) | [task-modes.md](docs/zh-CN/task-modes.md) |
| Add a new model | [add-model.md](docs/en/add-model.md) | [add-model.md](docs/zh-CN/add-model.md) |
| Add a new dataset | [add-dataset.md](docs/en/add-dataset.md) | [add-dataset.md](docs/zh-CN/add-dataset.md) |
| Traffic / spatiotemporal graphs | [datasets-traffic.md](docs/en/datasets-traffic.md) | [datasets-traffic.md](docs/zh-CN/datasets-traffic.md) |
| Pre-process datasets | [pre-process.md](docs/en/pre-process.md) | [pre-process.md](docs/zh-CN/pre-process.md) |
| Models reference | [models.md](docs/en/models.md) | [models.md](docs/zh-CN/models.md) |
| Visualize datasets | [visualize-data.md](docs/en/visualize-data.md) | [visualize-data.md](docs/zh-CN/visualize-data.md) |
| Dataset characteristics | [dataset-characteristics.md](docs/en/dataset-characteristics.md) | [dataset-characteristics.md](docs/zh-CN/dataset-characteristics.md) |
| Aggregate results | [aggregate-results.md](docs/en/aggregate-results.md) | [aggregate-results.md](docs/zh-CN/aggregate-results.md) |
| Model rankings | [rank-models.md](docs/en/rank-models.md) | [rank-models.md](docs/zh-CN/rank-models.md) |
| Bubble chart | [plot-bubble.md](docs/en/plot-bubble.md) | [plot-bubble.md](docs/zh-CN/plot-bubble.md) |
| GIFT-EVAL | [gift-eval.md](docs/en/gift-eval.md) | [gift-eval.md](docs/zh-CN/gift-eval.md) |
| Workflow scripts | [scripts.md](docs/en/scripts.md) | [scripts.md](docs/zh-CN/scripts.md) |
| Roadmap (deferred tasks) | [roadmap.md](docs/en/roadmap.md) | [roadmap.md](docs/zh-CN/roadmap.md) |
