# ModernTSF — 现代时间序列预测框架

结构清楚、工程化的时间序列预测实验框架。AI friendly、文档清晰、代码结构清楚，可用 TOML 组合快速跑复杂实验，支持性能分析与可视化展示。面向可复现实验、清晰的实验管理与快速迭代。

## 特点

- **TOML 组合配置**：数据集、模型、sweep 轻松编排复杂实验，配置清晰且便于版本管理
- **31 个开箱即用模型**：从简单线性基线到现代 Transformer、MLP 等
- **快速上手**：单配置、模型 sweep、数据集 sweep、多维 sweep，支持 `sweep.extend` 明确展开顺序
- **结果分析与可视化**：汇总指标、对比性能，一键绘制图表
- **AI friendly**：文档与代码结构清晰，为 VibeCode 带来巨大便利
- **扩展友好**：数据集、模型、指标可独立插拔，改动少、上手快

## 快速开始

创建环境并安装依赖：

```bash
uv sync --python 3.12
```

运行单数据集实验：

```bash
uv run modern-tsf --config configs/runs/run_single_data.toml
```

运行模型 sweep、数据集 sweep 或多维 sweep：

```bash
uv run modern-tsf --config configs/runs/sweep_model.toml
uv run modern-tsf --config configs/runs/sweep_data.toml
uv run modern-tsf --config configs/runs/multi_sweep.toml
```

`sweep.extend` 优先展开，再与其余 `[sweep]` 组合。总运行数为所有 extend 轴与 sweep 值的笛卡尔积。

汇总结果并绘制气泡图：

```bash
uv run python tool/aggregate_results.py --dataset ETTh1
uv run python tool/plot_bubble.py --csv work_dirs/ETTh1/results_all.csv --x mse --y mae --size total_params
```

按 pred_len/seed 输出模型排名：

```bash
uv run python tool/rank_models.py --dataset ETTh1
```

## 可用模型（31 个）

| 名称 | 类别 |
|---|---|
| `Linear`, `DLinear`, `NLinear`, `RLinear` | 线性基线 |
| `CrossLinear`, `MixLinear` | 线性变体 |
| `Autoformer`, `FEDformer`, `PatchTST`, `iTransformer` | Transformer 类 |
| `PatchMLP`, `xPatch`, `TSMixer`, `LightTS` | MLP / Patch 类 |
| `TimesNet` | CNN（二维时频域） |
| `TimeMixer` | 多尺度混合 |
| `SegRNN` | RNN（分段） |
| `FITS`, `SparseTSF`, `CycleNet`, `TiDE`, `SCINet` | 现代预测器 |
| `Amplifier`, `TimeBase`, `TimeBridge`, `TimeEmb` | 架构变体 |
| `PaiFilter`, `TexFilter` | 滤波类 |
| `SVTime`, `CMoS`, `PWS` | 其他 |

所有模型均有 `configs/models/` 下的 TOML 配置。模型参数定义在 `src/models/<name>/schema.py`。

## 可用数据集

| 配置 | 说明 |
|---|---|
| `configs/datasets/etth1.toml` | ETT 小时级数据 1 |
| `configs/datasets/etth2.toml` | ETT 小时级数据 2 |
| `configs/datasets/ettm1.toml` | ETT 分钟级数据 1 |
| `configs/datasets/ettm2.toml` | ETT 分钟级数据 2 |
| `configs/datasets/electricity.toml` | 电力消耗（321 通道） |
| `configs/datasets/weather.toml` | 气象多变量（21 通道） |
| `configs/datasets/traffic.toml` | 道路交通（862 通道） |
| `configs/datasets/solar.toml` | 太阳能（文本文件） |
| `configs/datasets/pre_processed.toml` | 预窗口化 `.npz` 文件 |

预切分与合成数据集（`periodic`、`trend`）同样受支持，详见 `docs/zh-CN/add-dataset.md`。

## 工具脚本

| 脚本 | 用途 |
|---|---|
| `tool/inspect_config.py` | 预览配置展开结果（运行数、数据集、模型） |
| `tool/aggregate_results.py` | 合并数据集的 performance + profile CSV |
| `tool/plot_bubble.py` | 从汇总 CSV 绘制气泡图 |
| `tool/rank_models.py` | 按 pred_len / seed 对模型排名 |
| `tool/visual_data.py` | 从 TOML 配置可视化数据集样本 |
| `tool/pre_process.py` | 将 CSV 转换为预窗口化 `.npz` 文件 |
| `tool/tsne_ecg.py` | ECG 数据集 t-SNE 对比可视化 |

常用工作流的 Shell 脚本位于 `scripts/` 目录。

## 文档（中文）

- 参数说明：`docs/zh-CN/params.md`
- 配置读取与使用：`docs/zh-CN/configs.md`
- 如何加入新模型：`docs/zh-CN/add-model.md`
- 如何加入新数据集：`docs/zh-CN/add-dataset.md`
- 数据集预处理：`docs/zh-CN/pre-process.md`
- 模型参考：`docs/zh-CN/models.md`
- 数据集可视化：`docs/zh-CN/visualize-data.md`
- 结果汇总：`docs/zh-CN/aggregate-results.md`
- 模型排名：`docs/zh-CN/rank-models.md`
- 气泡图：`docs/zh-CN/plot-bubble.md`
