<div align="center">

# 🚀 ModernTSF

**Modern Time Series Forecasting**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Time Series Forecasting](https://img.shields.io/badge/task-time%20series%20forecasting-blue.svg)](docs/en/models.md)
[![Models: 100+](https://img.shields.io/badge/models-100+-orange.svg)](docs/en/models.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Agent Infrastructure for time-series forecasting** — not just another toolkit.
A unified, reproducible substrate where humans and agents spend their time on the
*idea*, not the plumbing around it.

🗣️ **Clone the repo, open [Claude Code](https://claude.com/claude-code) or [Codex](https://developers.openai.com/codex), and speak your idea. That's the whole workflow.**

[**English**](README.md) | [**中文**](README_zh.md)

</div>

> 🧪 **Latest features land on the [`dev`](https://github.com/Diaugeia/ModernTSF/tree/dev) branch first.** `main` is the stable, versioned release line — if you want the newest (pre-release) capabilities, track or install from `dev`.

---

## 🧭 What is ModernTSF

You don't build a car to drive one, mill flour to bake a loaf, or grow your
own beans for a cup of coffee — you reach for something ready-made. AI
research needs the same layer: today's agents can write code and run
experiments, yet most of the effort — human and agent alike — still goes into
reproducing prior work, validating baselines, debugging environments, and
writing glue code. ModernTSF is that missing infrastructure layer for
time-series forecasting. You bring the idea; the substrate handles everything
around it.

---

## ✨ Highlights

- 🧠 **100+ forecasters, 60+ datasets** — linear baselines to Transformers, graph and foundation models; classic benchmarks, any custom CSV, traffic graphs, GIFT-EVAL
- 🤖 **Agent-first** — project instructions and 19 Agent Skills shared by Claude Code and Codex; every workflow is one sentence away
- 🎛️ **Three data settings** — `time_series`, `spatiotemporal`, and `covariate`, switchable per run
- 🔁 **Reproducible & auditable** — versioned TOML configs, fixed seeds, profiled outputs, and replayable agent trajectories make results genuinely comparable
- 🛠️ **One entry point** — `tsf` scaffolds, smoke-tests, sweeps, aggregates, ranks, plots, and reports

---

## 🏁 How to use

```bash
git clone https://github.com/Diaugeia/ModernTSF.git
cd ModernTSF
claude        # or: codex
```

Then just say what you want, in plain language:

```text
> Set up the environment for my GPU.
> Benchmark DLinear, PatchTST and iTransformer on ETTh1 and give me a leaderboard.
> Here is my CSV of hourly sales — add it as a dataset and find the best model for it.
> I have an idea: <describe it>. Scaffold a model, implement it, and compare it against strong baselines.
```

The agent picks up the project instructions and Agent Skills automatically —
environment setup, scaffolding, smoke tests, sweeps, aggregation, ranking, and
reports included.

---

## 📖 Documentation

The full reference lives in the docs index — parameters, configs, task modes,
adding models and datasets, tools, and the GIFT-EVAL benchmark:

🇬🇧 [English docs](docs/en/README.md) · 🇨🇳 [中文文档](docs/zh-CN/README.md)

> But chances are you'll never need any of this — let the agent do the reading.

---

## 📜 License

ModernTSF is released under the [MIT License](LICENSE) — open by default, free to
use, modify, and build upon.

Copyright © 2026 **Diaugeia.AI**.

Vendored third-party model implementations remain under their original upstream
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Diaugeia/ModernTSF&type=Date)](https://star-history.com/#Diaugeia/ModernTSF&Date)
