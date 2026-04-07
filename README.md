<div align="center">

# 🚀 ModernTSF

**Modern Time Series Forecasting**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Models: 31](https://img.shields.io/badge/models-31-orange.svg)](#-available-models-31)
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
- 🧠 **31 models out of the box** — from simple linear baselines to modern Transformers, MLPs, and more
- 📊 **60+ datasets** — 9 classic benchmarks + 53 GIFT-EVAL configurations across 23 domains and 10 frequencies
- ⚡ **Fast to run** — single configs, model sweeps, dataset sweeps, multi-axis sweeps, and explicit `sweep.extend` order
- 📈 **Profiling & visualization** — aggregate results, track metrics, and plot charts quickly
- 🤖 **AI-friendly** — clear docs and code structure that make VibeCode workflows fast and low-friction
- 🔌 **Extensible by design** — plug in new datasets, models, and metrics with minimal wiring

---

## 🏁 Quick Start

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

## 🧠 Available Models (31)

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
| `tool/pre_process.py` | Convert CSVs to pre-windowed `.npz` files |
| `tool/gift_eval_download.py` | Download GIFT-EVAL datasets + create symlink |

Shell scripts for common workflows are in `scripts/`.

---

## 📖 Documentation

- 🇬🇧 [English docs](docs/en/) — parameters, configs, add-model, add-dataset, tools
- 🇨🇳 [中文文档](docs/zh-CN/) — 参数、配置、添加模型、添加数据集、工具

| Topic | English | 中文 |
|---|---|---|
| Parameters reference | [params.md](docs/en/params.md) | [params.md](docs/zh-CN/params.md) |
| Config loading | [configs.md](docs/en/configs.md) | [configs.md](docs/zh-CN/configs.md) |
| Add a new model | [add-model.md](docs/en/add-model.md) | [add-model.md](docs/zh-CN/add-model.md) |
| Add a new dataset | [add-dataset.md](docs/en/add-dataset.md) | [add-dataset.md](docs/zh-CN/add-dataset.md) |
| Pre-process datasets | [pre-process.md](docs/en/pre-process.md) | [pre-process.md](docs/zh-CN/pre-process.md) |
| Models reference | [models.md](docs/en/models.md) | [models.md](docs/zh-CN/models.md) |
| Visualize datasets | [visualize-data.md](docs/en/visualize-data.md) | [visualize-data.md](docs/zh-CN/visualize-data.md) |
| Aggregate results | [aggregate-results.md](docs/en/aggregate-results.md) | [aggregate-results.md](docs/zh-CN/aggregate-results.md) |
| Model rankings | [rank-models.md](docs/en/rank-models.md) | [rank-models.md](docs/zh-CN/rank-models.md) |
| Bubble chart | [plot-bubble.md](docs/en/plot-bubble.md) | [plot-bubble.md](docs/zh-CN/plot-bubble.md) |
