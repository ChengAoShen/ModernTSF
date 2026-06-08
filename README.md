<div align="center">

# 🚀 ModernTSF

**Modern Time Series Forecasting**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Time Series Forecasting](https://img.shields.io/badge/task-time%20series%20forecasting-blue.svg)](#-available-models-100)
[![Models: 100+](https://img.shields.io/badge/models-100+-orange.svg)](#-available-models-100)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Agent Infrastructure for time-series forecasting** — not just another toolkit.
A unified, reproducible substrate where humans and agents spend their time on the
*idea*, not the plumbing around it.

[**English**](README.md) | [**中文**](README_zh.md)

</div>

---

## 🧭 Why ModernTSF

You don't build a car to drive one, and you don't mix reagents from scratch to run
a biology experiment — you reach for a kit. AI research needs the same layer.
Today's agents are strong: they can gather information, write code, and run
experiments. Yet for agents and humans alike, most of that effort never touches the
core idea — it is spent searching for and reproducing prior work, validating
baselines on your own data, debugging environments, and writing glue code. The way
research gets done is shifting in stages: **pure human** work is exhausting and
swamped by peripheral labor; **human + agent** raises the ceiling but only moves
the bottleneck, so now the time goes into babysitting agents, waiting on them, and
shuttling their output around. The next step is **human + agent + agent
infrastructure** — the human contributes the simplest, most creative idea, the
agent spends its compute on the *core component*, and the infrastructure absorbs
everything around it.

ModernTSF is that missing infrastructure layer for time-series forecasting. It
keeps human and agent effort on the most transformative part of a problem instead
of disposable surrounding code, and it turns a pile of individual repos that can't
be verified against each other into one shared, fair benchmark: by recording an
agent's full behavioral trajectory, every step of an experiment can be replayed,
so results stay openly auditable and genuinely comparable. With 100+ built-in
forecasters it doubles as a framework for learning the field — a fast way for
newcomers and agents to grasp which models exist and what makes each one distinct.

In practice that means a new idea can be checked for originality and situated
against what it builds on within minutes; baselines are auto-selected for
comparison, the environment is set up for you, and the burden of experimental rigor
is lifted off your shoulders. Everything below — docs-first code, structured TOML
configs, and Agent Skills — exists so that agents (and the people steering them) can
operate with as little friction as possible.

---

## ✨ Highlights

- 📝 **TOML-first configs** — compose datasets, models, and sweeps into complex, fully versionable experiments
- 🧠 **100+ models out of the box** — across the `time_series`, `spatiotemporal`, and `covariate` settings, from linear baselines and Transformers to graph and foundation models
- 🎛️ **Three forecasting data settings** — `time_series`, `spatiotemporal`, and `covariate`, switchable per run
- 📊 **60+ datasets** — classic benchmarks, any custom CSV, traffic graphs (METR-LA, PEMS0x), node-structured air-quality, and the 53-config GIFT-EVAL benchmark
- ⚡ **Fast to run** — single configs, model / dataset / multi-axis sweeps, with explicit `sweep.extend` ordering
- 🎚️ **Rich metrics, losses & training tricks** — `mse`/`mae`/`rmse`/`mape`/`mspe`/`corr`/`rse`/`wape`/`smape` (`mase` opt-in), masked losses, `[training.tricks]` (`grad_clip`/`grad_accum`/`curriculum` + model aux-loss), rolling evaluation, and graph adjacency normalization
- 📈 **Profiling & visualization** — aggregate results, rank models, and plot charts in one step
- 🔁 **Reproducible by construction** — versioned configs, fixed seeds, profiled CSV outputs, and replayable agent trajectories
- 🤖 **Built for agents** — docs-first code, structured configs, and Agent Skills keep VibeCode workflows fast and low-friction
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

## 📦 What's Inside

The full catalogs live in the docs — the README stays focused on getting you running.

| Area | At a glance | Reference |
|---|---|---|
| 🧠 **Models** | 100+ forecasters across `time_series`, `spatiotemporal`, and `covariate` settings (Transformers, MLP/patch, CNN/RNN, modern forecasters, graph/spatiotemporal, classical-ML adapters, foundation models) | [models.md](docs/en/models.md) |
| 📊 **Datasets** | 60+ configs — classic benchmarks, any custom CSV, traffic graphs (METR-LA, PEMS0x), node-structured air-quality, and the 53-config GIFT-EVAL benchmark | [add-dataset.md](docs/en/add-dataset.md) · [gift-eval.md](docs/en/gift-eval.md) |
| 🛠️ **Tools (`tsf`)** | One entry point for scaffolding, smoke-testing, running sweeps, aggregating, ranking, and plotting | [scripts.md](docs/en/scripts.md) |
| 🤖 **Agent Skills** | `.claude/skills/` wrap every tool for agent/human use via `/<name>` | [docs index](docs/en/README.md) |

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

---

## 📜 License

ModernTSF is released under the [MIT License](LICENSE) — open by default, free to
use, modify, and build upon.

Copyright © 2026 **Diaugeia.AI**.

Vendored third-party model implementations remain under their original upstream
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
