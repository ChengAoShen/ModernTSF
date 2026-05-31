<div align="center">

# 🚀 ModernTSF

**现代时间序列预测框架**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Models: 99](https://img.shields.io/badge/模型-99-orange.svg)](#-内置模型-99)
[![Datasets: 60+](https://img.shields.io/badge/数据集-60+-purple.svg)](#-支持的数据集)
[![GIFT-EVAL](https://img.shields.io/badge/GIFT--EVAL-53_配置-blueviolet.svg)](#-gift-eval-基准测试)

结构化、工程级的时间序列预测基准框架。
AI 友好、文档优先、易于扩展 — 通过 TOML 配置组合、性能分析和丰富的可视化，
快速运行复杂实验。

[**English**](README.md) | [**中文**](README_zh.md)

</div>

---

## 🧭 设计理念

ModernTSF 围绕四项承诺构建：

- ✨ **Modern（现代）** — 基于前沿架构构建，并持续更新，始终保持在时间序列预测的最前沿。
- 🤖 **Agentic（智能体友好）** — 从设计之初就面向 LLM agent（文档优先的代码、Agent Skills、结构化配置），让人类减少手动接线工作。
- 🔁 **Reproducible（可复现）** — 每个结果都可追溯、可重跑、可验证：可版本化的 TOML 配置、固定随机种子、带性能分析的 CSV 输出。
- 🔓 **Open by default（开放优先）** — 透明、可审计、可自由二次开发，采用宽松的 MIT 许可证。

---

## ✨ 特性

- 📝 **TOML 配置驱动** — 通过清晰、可版本化的配置文件组合数据集、模型和扫描实验
- 🧠 **99 个开箱即用的模型** — 从简单线性基线到 Transformer、MLP、时空与空气质量等现代架构
- 📊 **60+ 数据集** — 9 个经典基准 + 自定义 CSV（exchange、ili …）+ 交通图（METR-LA、PEMS0x）+ 53 个 GIFT-EVAL 配置，覆盖 23 个领域和 10 种频率
- ⚡ **高效运行** — 单配置、模型扫描、数据集扫描、多轴扫描，支持 `sweep.extend` 显式排列
- 🎚️ **指标、损失与训练技巧** — `mse`/`mae`/`rmse`/`mape`/`mspe`/`corr`/`rse`/`wape`/`smape`（`mase` 可选），掩码损失（`masked_mae`/`mse`/`rmse`），`[training.tricks]` 的 `grad_clip`/`grad_accum`/`curriculum`（+ 模型辅助损失），`[evaluation] strategy="rolling"`，以及图邻接归一化 `[dataset.params] adj_norm`
- 📈 **性能分析与可视化** — 聚合结果、追踪指标、快速绘图
- 🤖 **AI 友好** — 清晰的文档和代码结构，让 VibeCode 工作流快速顺畅
- 🔌 **可扩展设计** — 用最少的代码接入新数据集、模型和评估指标

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

## 🧠 内置模型 (99)

| 类别 | 数量 | 模型 |
|---|---|---|
| 线性（Linear） | 6 | `Linear`, `DLinear`, `NLinear`, `RLinear`, `CrossLinear`, `MixLinear` |
| Transformer 系列 | 21 | `PatchTST`, `iTransformer`, `TimeXer`, `Crossformer`, `Informer`, `Autoformer`, `FEDformer`, `Reformer`, `Pyraformer`, `ETSformer`, `NSTransformer`, `MultiPatchFormer`, `PAttn`, `CARD`, `Fredformer`, `DUET`, `Pathformer`, `DSFormer`, `DTAF`, `TimePerceiver`, `Transformer` |
| MLP / 补丁方法 | 11 | `PatchMLP`, `xPatch`, `TSMixer`, `LightTS`, `WPMixer`, `MTSMixer`, `UMixer`, `NHiTS`, `NBeats`, `HDMixer`, `SRSNet` |
| CNN | 5 | `TimesNet`, `SCINet`, `MICN`, `ModernTCN`, `WaveNet` |
| RNN | 6 | `SegRNN`, `DeepAR`, `MambaSimple`, `S_Mamba`, `BiMamba`, `S4` |
| 现代预测器 | 10 | `TimeMixer`, `FITS`, `SparseTSF`, `CycleNet`, `TiDE`, `FiLM`, `FreTS`, `Koopa`, `SOFTS`, `TimeKAN` |
| 架构变体 | 4 | `Amplifier`, `TimeBase`, `TimeBridge`, `TimeEmb` |
| 滤波器方法 | 2 | `PaiFilter`, `TexFilter` |
| 其他 | 7 | `SVTime`, `CMoS`, `PWS`, `Sumba`, `CrossGNN`, `MSGNet`, `TimeFilter` |
| 图 / 时空（Tier 2） | 20 | `STID`, `GWNet`, `STGCN`, `DCRNN`, `MTGNN`, `AGCRN`, `STNorm`, `StemGNN`, `STGODE`, `STAEformer`, `GTS`, `DGCRN`, `STDN`, `DFDGCN`, `STPGNN`, `D2STGNN`, `MegaCRN`, `HimNet`, `BigST`, `STWave` |
| 移植自 PoorOtterBob | 7 | `MoFo`, `PHAT`, `BiST`, `MAGE`, `STOP`, `CauAir`, `AirCade` |

移植来源：[Time-Series-Library](https://github.com/thuml/Time-Series-Library)（MIT）、
[BasicTS](https://github.com/GestaltCogTeam/BasicTS)（Apache-2.0）、TFB 与
[PoorOtterBob](https://github.com/PoorOtterBob)。所有模型的 TOML 配置在
`configs/models/`，参数定义在 `src/models/<name>/schema.py`。完整的逐模型表见
`docs/zh-CN/models.md`。

最后七个模型移植自 [PoorOtterBob](https://github.com/PoorOtterBob) 系列仓库，在此作为标准单变量通道预测器运行。其适配器将 ModernTSF 的 `(x_enc, x_mark_enc, x_dec, x_mark_dec)` batch 转换为各模型的原生布局——见 `src/models/_external/marks.py`。时空与空气质量模型接收形状为 `(B, T, N, 1+F)` 的张量，值通道额外拼接 `F = 2` 个归一化日历特征（time-of-day、day-of-week）；空气质量模型还会把未来日历特征作为协变量。`PHAT` 的上游仓库缺失其核心 `PHAT_Attention` 模块，已依据论文（ICLR 2026, arXiv:2602.00654）在 `src/models/phat/layers/PHAT_Attention.py` 中重建。AirCade 使用频域 MAE（`loss = "freq_mae"`）训练，其余默认 MAE。每个模型的端到端 smoke run 见 `configs/runs/smoke_*.toml`（参见 `scripts/make_smoke_data.py`）。

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

### 自定义 CSV 数据集

任意扁平多变量 CSV 都可通过 `name = "custom"` 接入 `Dataset_Custom` —— 只需配置，无需新代码。内置示例：`exchange`、`ili`、`nn5`、`fred_md`、`beijing_air`、`aqshunyi`、`aqwan`（见 `configs/datasets/*.toml`）。

### 交通图数据集

节点 + 邻接的交通数据包复用 `cauair_st` 节点加载器：`metr_la`、`pems_bay`、`pems03`、`pems04`、`pems07`、`pems08`。可用 `tool/convert_traffic.py` 从原始数组构建数据包；详见 [datasets-traffic.md](docs/zh-CN/datasets-traffic.md)。

### 时空与空气质量

面向 `spatiotemporal` 和 `covariate` 任务模式的节点结构数据集（见 [任务模式](#-任务模式)）：

| 配置文件 | 说明 |
|---|---|
| `configs/datasets/synthetic_st.toml` | 合成节点序列，带日历协变量 `[time_in_day, day_in_week]` |
| `configs/datasets/cauair_ccaq_st.toml` | CauAir / CCAQ 空气质量（209 节点，气象协变量）—— 时空布局 |
| `configs/datasets/cauair_ccaq_ts.toml` | 同样的 CauAir 数据作为普通预测数据集（节点 → 通道） |

CauAir 的 `.npz` 包（`his.npz`、`idx_{train,val,test}.npy`、`adj_mx.npy`）由 `cauair_st` / `cauair_ts` 加载，放置于 `dataset/<name>/` 下。交通图数据包（METR-LA / PEMS-BAY / PEMS0x）复用 `cauair_st` 节点加载器，详见 `docs/zh-CN/datasets-traffic.md`。

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
| `tool/visualize_predictions.py` | 为已训练的 run 绘制预测值对真实值的 case 图 |
| `tool/dataset_characteristics.py` | 提取 TFB 风格的数据集特征（趋势 / 季节性 / 平稳性） |
| `tool/convert_traffic.py` | 为 `cauair_st` 构建交通 / 时空节点数据包（值数组 + 邻接矩阵） |
| `tool/pre_process.py` | 将 CSV 转为预切窗 `.npz` 文件 |
| `tool/gift_eval_download.py` | 下载 GIFT-EVAL 数据集 + 创建软链接 |

### 工作流脚本

`scripts/` 下提供了参数化的常用工作流封装脚本 — 位置参数 + 环境变量覆盖（`-h`/`--help` 打印用法）：

```bash
# 顺序运行一个或多个配置（默认 configs/runs/run_single_data.toml）
[GPU_IDS=<ids>] bash scripts/run_multi_configs.sh [config ...]

# 聚合某数据集/pred_len 后绘制气泡图
[DATASET=… PRED_LEN=… X=… Y=… SIZE=… OUT_CSV=… OUT_SVG=…] \
  bash scripts/aggregate_and_plot.sh [DATASET] [PRED_LEN]
```

`run_multi_configs.sh` 接受任意数量的 TOML 配置路径（环境变量 `GPU_IDS`，默认 `0`）。`aggregate_and_plot.sh` 接受位置参数 `DATASET`（默认 `ETTh1`）和 `PRED_LEN`（默认 `96`），二者均可由同名环境变量覆盖，另支持 `X`/`Y`/`SIZE` 和 `OUT_CSV`/`OUT_SVG` 环境变量覆盖。`detect_hardware.sh` 报告 GPU/CUDA 并推荐 `UV_TORCH_BACKEND` 标签。详见 [scripts.md](docs/zh-CN/scripts.md)。

### 🤖 Agent Skills

本仓库在 `.claude/skills/` 下附带 [Claude Code](https://claude.ai/code) Skills — `setup-env`、`run`、`experiments`、`aggregate`、`visualize`、`characteristics`、`pre-process`、`add-dataset`、`add-model`、`inspect`、`rank`、`plot`、`gift-eval`、`sweep` — 封装上述工具，供 agent 或人类通过 `/<name>` 使用。

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
