<div align="center">

# 🚀 ModernTSF

**现代时间序列预测框架**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Models: 31](https://img.shields.io/badge/模型-31-orange.svg)](#-内置模型-31)
[![Datasets: 60+](https://img.shields.io/badge/数据集-60+-purple.svg)](#-支持的数据集)
[![GIFT-EVAL](https://img.shields.io/badge/GIFT--EVAL-53_配置-blueviolet.svg)](#-gift-eval-基准测试)

结构化、工程级的时间序列预测基准框架。
AI 友好、文档优先、易于扩展 — 通过 TOML 配置组合、性能分析和丰富的可视化，
快速运行复杂实验。

[**English**](README.md) | [**中文**](README_zh.md)

</div>

---

## ✨ 特性

- 📝 **TOML 配置驱动** — 通过清晰、可版本化的配置文件组合数据集、模型和扫描实验
- 🧠 **31 个开箱即用的模型** — 从简单线性基线到 Transformer、MLP 等现代架构
- 📊 **60+ 数据集** — 9 个经典基准 + 53 个 GIFT-EVAL 配置，覆盖 23 个领域和 10 种频率
- ⚡ **高效运行** — 单配置、模型扫描、数据集扫描、多轴扫描，支持 `sweep.extend` 显式排列
- 📈 **性能分析与可视化** — 聚合结果、追踪指标、快速绘图
- 🤖 **AI 友好** — 清晰的文档和代码结构，让 VibeCode 工作流快速顺畅
- 🔌 **可扩展设计** — 用最少的代码接入新数据集、模型和评估指标

---

## 🏁 快速开始

创建环境并安装依赖：

```bash
uv sync --python 3.12
```

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

## 🧠 内置模型 (31)

| 名称 | 类别 |
|---|---|
| `Linear`, `DLinear`, `NLinear`, `RLinear` | 线性基线 |
| `CrossLinear`, `MixLinear` | 线性变体 |
| `Autoformer`, `FEDformer`, `PatchTST`, `iTransformer` | Transformer 系列 |
| `PatchMLP`, `xPatch`, `TSMixer`, `LightTS` | MLP /补丁方法 |
| `TimesNet` | CNN（2D 时频） |
| `TimeMixer` | 多尺度混合 |
| `SegRNN` | RNN（分段式） |
| `FITS`, `SparseTSF`, `CycleNet`, `TiDE`, `SCINet` | 现代预测器 |
| `Amplifier`, `TimeBase`, `TimeBridge`, `TimeEmb` | 架构变体 |
| `PaiFilter`, `TexFilter` | 滤波器方法 |
| `SVTime`, `CMoS`, `PWS` | 其他 |

所有模型的 TOML 配置在 `configs/models/`，模型参数定义在 `src/models/<name>/schema.py`。

---

## 📊 支持的数据集

### 经典基准

| 配置文件 | 说明 |
|---|---|
| `configs/datasets/etth1.toml` | ETT 小时级 1 |
| `configs/datasets/etth2.toml` | ETT 小时级 2 |
| `configs/datasets/ettm1.toml` | ETT 分钟级 1 |
| `configs/datasets/ettm2.toml` | ETT 分钟级 2 |
| `configs/datasets/electricity.toml` | 电力消耗（321 通道） |
| `configs/datasets/weather.toml` | 气象多变量（21 通道） |
| `configs/datasets/traffic.toml` | 道路交通（862 通道） |
| `configs/datasets/solar.toml` | 太阳能发电 |
| `configs/datasets/pre_processed.toml` | 预切窗 `.npz` 文件 |

预拆分和合成数据集（`periodic`、`trend`）也受支持 — 详见 `docs/zh-CN/add-dataset.md`。

### 🏆 GIFT-EVAL 基准测试

ModernTSF 原生支持 [**GIFT-EVAL**](https://huggingface.co/datasets/Salesforce/GiftEval) 基准 — **53 个数据集配置**，覆盖 **23 个基础数据集**、**10 种频率**（从秒级到月级）和 **7 个领域**（能源、交通、气象、金融等）。

<details>
<summary><b>📋 完整 GIFT-EVAL 数据集列表（点击展开）</b></summary>

| 数据集 | 频率 | 类型 |
|---|---|---|
| electricity | 15T, D, H, W | 单变量 |
| ett1, ett2 | 15T, D, H, W | 多变量（7 维） |
| solar | 10T, D, H, W | 单变量 |
| LOOP_SEATTLE | 5T, D, H | 单变量 |
| jena_weather | 10T | 多变量（21 维） |
| M_DENSE | D, H | 单变量 |
| SZ_TAXI | 15T, H | 单变量 |
| bitbrains_fast_storage | 5T, H | 多变量（2 维） |
| bitbrains_rnd | 5T, H | 多变量（2 维） |
| bizitobs_application | 10S | 多变量（2 维） |
| bizitobs_l2c | 5T, H | 多变量（7 维） |
| bizitobs_service | 10S | 多变量（2 维） |
| hierarchical_sales | D, W | 单变量 |
| kdd_cup_2018_with_missing | D, H | 单变量 |
| saugeenday | D, M, W | 单变量 |
| us_births | D, M, W | 单变量 |
| m4_daily, m4_hourly, m4_monthly | — | 单变量 |
| m4_quarterly, m4_weekly, m4_yearly | — | 单变量 |
| car_parts_with_missing | M | 单变量 |
| covid_deaths | D | 单变量 |
| hospital | M | 单变量 |
| restaurant | D | 单变量 |
| temperature_rain_with_missing | D | 单变量 |

</details>

**快速使用：**

```bash
# 下载全部 GIFT-EVAL 数据集（自选存储位置）
uv run python tool/gift_eval_download.py --output-dir /your/path

# 或链接已有数据
uv run python tool/gift_eval_download.py --link-only --output-dir /path/to/GiftEval

# 运行完整 GIFT-EVAL 扫描（short term，全部 53 个数据集）
uv run modern-tsf --config configs/runs/gift_eval_sweep.toml
```

每个数据集 TOML 默认使用 GIFT-EVAL **short-term** 预测长度。medium（10x）和 long（15x）的值已标注在各配置文件中 — 修改 `pred_len` 即可切换。

---

## 🛠️ 工具

| 脚本 | 用途 |
|---|---|
| `tool/inspect_config.py` | 预览配置展开（扫描数、数据集、模型） |
| `tool/aggregate_results.py` | 聚合某数据集的性能 + profile CSV |
| `tool/plot_bubble.py` | 从聚合 CSV 绘制气泡图 |
| `tool/rank_models.py` | 按 pred_len / seed 排名模型 |
| `tool/visual_data.py` | 从 TOML 配置可视化数据集样本 |
| `tool/pre_process.py` | 将 CSV 转为预切窗 `.npz` 文件 |
| `tool/gift_eval_download.py` | 下载 GIFT-EVAL 数据集 + 创建软链接 |

常用工作流脚本在 `scripts/` 目录下。

---

## 📖 文档

- 🇬🇧 [English docs](docs/en/) — parameters, configs, add-model, add-dataset, tools
- 🇨🇳 [中文文档](docs/zh-CN/) — 参数、配置、添加模型、添加数据集、工具

| 主题 | English | 中文 |
|---|---|---|
| 参数参考 | [params.md](docs/en/params.md) | [params.md](docs/zh-CN/params.md) |
| 配置加载 | [configs.md](docs/en/configs.md) | [configs.md](docs/zh-CN/configs.md) |
| 添加新模型 | [add-model.md](docs/en/add-model.md) | [add-model.md](docs/zh-CN/add-model.md) |
| 添加新数据集 | [add-dataset.md](docs/en/add-dataset.md) | [add-dataset.md](docs/zh-CN/add-dataset.md) |
| 预处理数据集 | [pre-process.md](docs/en/pre-process.md) | [pre-process.md](docs/zh-CN/pre-process.md) |
| 模型参考 | [models.md](docs/en/models.md) | [models.md](docs/zh-CN/models.md) |
| 可视化数据集 | [visualize-data.md](docs/en/visualize-data.md) | [visualize-data.md](docs/zh-CN/visualize-data.md) |
| 聚合结果 | [aggregate-results.md](docs/en/aggregate-results.md) | [aggregate-results.md](docs/zh-CN/aggregate-results.md) |
| 模型排名 | [rank-models.md](docs/en/rank-models.md) | [rank-models.md](docs/zh-CN/rank-models.md) |
| 气泡图 | [plot-bubble.md](docs/en/plot-bubble.md) | [plot-bubble.md](docs/zh-CN/plot-bubble.md) |
