<div align="center">

# 🚀 ModernTSF

**现代时间序列预测框架**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Time Series Forecasting](https://img.shields.io/badge/任务-时序预测-blue.svg)](#-内置模型-100)
[![Models: 100+](https://img.shields.io/badge/模型-100+-orange.svg)](#-内置模型-100)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**面向时间序列预测的 AI Infrastructure** —— 而不只是又一个工具包。
一个统一、可复现的底座，让人和 Agent 都把时间花在*创新 idea* 上，
而不是它周围的各种胶水工作。

[**English**](README.md) | [**中文**](README_zh.md)

</div>

---

## 🧭 为什么需要 ModernTSF

开车不必从造车开始，做生物实验也不必从头调配试剂——直接用试剂盒（kit）就好。
AI 研究同样需要这样一层基础设施。如今的 Agent 已经很强：能收集信息、写代码、跑实验。
但无论对 Agent 还是人类，这些精力大多并未触及核心 idea——而是耗在了搜索与复现已有
工作、在自己的数据集上验证 baseline、调试环境、编写周边胶水代码上。研究的方式正在
分阶段演进：**纯人类**方式过于疲惫，被无意义的周边劳动淹没；**人类 + Agent** 提高了
上限，却只是让瓶颈发生了转移——如今大量时间花在调试 Agent、等待 Agent、搬运 Agent
产出上。下一步是 **人类 + Agent + Agent Infrastructure**：人类只贡献最简洁、最具创新性
的 idea，Agent 把算力集中在实现*核心组件*上，其余一切由基础设施兜底。

ModernTSF 就是时间序列预测领域所缺失的这一层基础设施。它把人和 Agent 的时间集中在
问题最具变革性的部分，而不是用完即弃的周边代码上；并把一堆彼此无法验证的个人 repo，
变成一个共享、公平的基准——通过记录 Agent 完整的行为轨迹，实验的每一步都可被精确还原，
使结果公开可审计、真正可比。内置 100+ 预测器也让它同时成为一个学习该领域的框架，
让初学者和 Agent 都能快速建立对「有哪些模型、各自有何特点」的认知。

落到实处，这意味着：一个新 idea 能在几分钟内被判定是否原创、是在哪些基础上做的改进；
baseline 会被自动挑选用于对比，环境会被自动配置好，实验严谨性的负担从你肩上卸下。
下文的一切——文档优先的代码、结构化 TOML 配置、Agent Skills——都是为了让 Agent
（以及驾驭它们的人）以尽可能小的摩擦运作。

---

## ✨ 特性

- 📝 **TOML 配置驱动** — 用清晰、可版本化的配置组合数据集、模型与扫描，搭建复杂实验
- 🧠 **100+ 个开箱即用的模型** — 覆盖 `time_series`、`spatiotemporal`、`covariate` 三种设定，从线性基线、Transformer 到图模型与基础模型
- 🎛️ **三种预测数据设定** — `time_series`、`spatiotemporal`、`covariate`，可按 run 切换
- 📊 **60+ 数据集** — 经典基准、任意自定义 CSV、交通图（METR-LA、PEMS0x）、节点结构空气质量，以及 53 配置的 GIFT-EVAL 基准
- ⚡ **高效运行** — 单配置、模型 / 数据集 / 多轴扫描，支持 `sweep.extend` 显式排列
- 🎚️ **丰富的指标、损失与训练技巧** — `mse`/`mae`/`rmse`/`mape`/`mspe`/`corr`/`rse`/`wape`/`smape`（`mase` 可选）、掩码损失、`[training.tricks]`（`grad_clip`/`grad_accum`/`curriculum` + 模型辅助损失）、滚动评估、图邻接归一化
- 📈 **性能分析与可视化** — 一步完成聚合结果、模型排名与绘图
- 🔁 **天然可复现** — 可版本化配置、固定随机种子、带性能分析的 CSV 输出，以及可回放的 Agent 轨迹
- 🤖 **为 Agent 而生** — 文档优先的代码、结构化配置与 Agent Skills，让 VibeCode 工作流快速顺畅
- 🔌 **可扩展设计** — 用最少的代码接入新数据集、模型与评估指标

---

## 🎛️ 任务模式

所有任务都是**预测（forecasting）**；`task.mode` 选择数据设定。默认是 `time_series`，因此已有配置不受影响。

| 模式 | Batch | 目标 | 示例 |
|---|---|---|---|
| `time_series` | `(B, T, C)` 值 | 所有通道 | 任意 CSV 数据集 |
| `spatiotemporal` | `(B, T, N, 1+F)` 值 + 逐节点协变量 | `N` 个节点的值 | `synthetic_st`、`cauair_st` |
| `covariate` | spatiotemporal + **未来**协变量 | `N` 个节点的值 | `cauair_st` |

详见 `docs/zh-CN/task-modes.md`（或 `docs/en/task-modes.md`），含模型/模式兼容性说明。

---

## 🏁 快速开始

创建环境并安装依赖。PyTorch 构建（CPU 或某个 CUDA `cuXXX`）在安装时选择——
让 uv 自动探测你的 GPU：

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12   # 或 cu124 / cu121 / cpu …
```

> 新机器或新 GPU？运行 `bash scripts/detect_hardware.sh` 探测 CUDA 后端，
> 或使用 `setup-env` skill。详见 [setup-env.md](docs/zh-CN/setup-env.md)。

运行单数据集实验：

```bash
uv run modern-tsf --config configs/runs/run_single_data.toml
```

运行模型扫描、数据集扫描或多轴扫描：

```bash
uv run modern-tsf --config configs/runs/sweep_model.toml
uv run modern-tsf --config configs/runs/sweep_data.toml
uv run modern-tsf --config configs/runs/multi_sweep.toml
```

> `sweep.extend` 先展开，再与 `[sweep]` 的键做笛卡尔积。总运行数 = 所有 extend 轴 × 所有 sweep 值。

聚合结果并绘制气泡图：

```bash
uv run python tool/aggregate_results.py --dataset ETTh1
uv run python tool/plot_bubble.py --csv work_dirs/ETTh1/results_all.csv --x mse --y mae --size total_params
```

排名模型（按 `pred_len` / seed）：

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

---

## 📦 仓库内容一览

完整清单都在文档里 —— README 只保留让你快速跑起来的部分。

| 板块 | 概览 | 参考 |
|---|---|---|
| 🧠 **模型** | 100+ 预测器，覆盖 `time_series`、`spatiotemporal`、`covariate` 三种设定（Transformer、MLP/Patch、CNN/RNN、现代预测器、图/时空、传统 ML 适配器、基础模型） | [models.md](docs/zh-CN/models.md) |
| 📊 **数据集** | 60+ 配置 —— 经典基准、任意自定义 CSV、交通图（METR-LA、PEMS0x）、节点结构空气质量，以及 53 配置的 GIFT-EVAL 基准 | [add-dataset.md](docs/zh-CN/add-dataset.md) · [gift-eval.md](docs/zh-CN/gift-eval.md) |
| 🛠️ **工具（`tsf`）** | 一个入口完成脚手架、smoke 测试、运行扫描、聚合、排名、绘图 | [scripts.md](docs/zh-CN/scripts.md) |
| 🤖 **Agent Skills** | `.claude/skills/` 封装每个工具，供 agent/人类通过 `/<name>` 使用 | [文档索引](docs/zh-CN/README.md) |

---

## 📖 文档

- 🇬🇧 [English docs](docs/en/README.md) — parameters, configs, add-model, add-dataset, tools
- 🇨🇳 [中文文档](docs/zh-CN/README.md) — 参数、配置、添加模型、添加数据集、工具

| 主题 | English | 中文 |
|---|---|---|
| 环境配置（GPU/CUDA） | [setup-env.md](docs/en/setup-env.md) | [setup-env.md](docs/zh-CN/setup-env.md) |
| 参数参考 | [params.md](docs/en/params.md) | [params.md](docs/zh-CN/params.md) |
| 配置加载 | [configs.md](docs/en/configs.md) | [configs.md](docs/zh-CN/configs.md) |
| 一键实验 | [experiments.md](docs/en/experiments.md) | [experiments.md](docs/zh-CN/experiments.md) |
| 检查配置 | [inspect-config.md](docs/en/inspect-config.md) | [inspect-config.md](docs/zh-CN/inspect-config.md) |
| 任务模式 | [task-modes.md](docs/en/task-modes.md) | [task-modes.md](docs/zh-CN/task-modes.md) |
| 添加新模型 | [add-model.md](docs/en/add-model.md) | [add-model.md](docs/zh-CN/add-model.md) |
| 添加新数据集 | [add-dataset.md](docs/en/add-dataset.md) | [add-dataset.md](docs/zh-CN/add-dataset.md) |
| 交通 / 时空图 | [datasets-traffic.md](docs/en/datasets-traffic.md) | [datasets-traffic.md](docs/zh-CN/datasets-traffic.md) |
| 预处理数据集 | [pre-process.md](docs/en/pre-process.md) | [pre-process.md](docs/zh-CN/pre-process.md) |
| 模型参考 | [models.md](docs/en/models.md) | [models.md](docs/zh-CN/models.md) |
| 可视化数据集 | [visualize-data.md](docs/en/visualize-data.md) | [visualize-data.md](docs/zh-CN/visualize-data.md) |
| 数据集特征 | [dataset-characteristics.md](docs/en/dataset-characteristics.md) | [dataset-characteristics.md](docs/zh-CN/dataset-characteristics.md) |
| 聚合结果 | [aggregate-results.md](docs/en/aggregate-results.md) | [aggregate-results.md](docs/zh-CN/aggregate-results.md) |
| 模型排名 | [rank-models.md](docs/en/rank-models.md) | [rank-models.md](docs/zh-CN/rank-models.md) |
| 气泡图 | [plot-bubble.md](docs/en/plot-bubble.md) | [plot-bubble.md](docs/zh-CN/plot-bubble.md) |
| GIFT-EVAL | [gift-eval.md](docs/en/gift-eval.md) | [gift-eval.md](docs/zh-CN/gift-eval.md) |
| 工作流脚本 | [scripts.md](docs/en/scripts.md) | [scripts.md](docs/zh-CN/scripts.md) |
| 路线图（延后任务） | [roadmap.md](docs/en/roadmap.md) | [roadmap.md](docs/zh-CN/roadmap.md) |

---

## 📜 许可证

ModernTSF 以 [MIT 许可证](LICENSE) 发布 — 开放优先，可自由使用、修改与二次开发。

版权所有 © 2026 **Diaugeia.AI**。

仓库内置的第三方模型实现仍遵循其各自的上游许可证；归属信息见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
