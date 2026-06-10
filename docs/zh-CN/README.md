# 文档索引

← [返回项目根目录](../../README.md)

## 环境配置

| 文档 | 说明 |
|------|------|
| [setup-env.md](setup-env.md) | 检测本机 GPU/CUDA，并通过 `UV_TORCH_BACKEND` 用 uv 安装匹配的 PyTorch 后端。 |

## 参考手册

| 文档 | 说明 |
|------|------|
| [params.md](params.md) | 所有 TOML 字段含义，对应 `configs/base.toml` 和 Pydantic schema。 |
| [configs.md](configs.md) | 配置加载流程：`extends` 继承、`[sweep]` 展开与校验。 |
| [models.md](models.md) | 172 个可用模型的目录，含架构说明与关键超参数。 |
| [task-modes.md](task-modes.md) | 由 `task.mode` 选择的数据设定：`time_series`、`spatiotemporal`、`covariate`。 |

## 操作指南

| 文档 | 说明 |
|------|------|
| [add-model.md](add-model.md) | 添加新模型的分步指南：包结构、schema、注册表条目及 TOML 配置。 |
| [add-dataset.md](add-dataset.md) | 添加新数据集的分步指南：单文件、预分割和预处理三种方式。 |
| [datasets-traffic.md](datasets-traffic.md) | 转换并运行 METR-LA / PEMS-BAY / PEMS0x 交通图数据包（经 `cauair_st` 节点加载器）。 |
| [pre-process.md](pre-process.md) | 使用 `tool/pre_process.py` 将 CSV 预切片为 `.npz` 文件，供 `pre_processed` 数据集使用。 |
| [experiments.md](experiments.md) | 一键实验：启动 sweep、聚合、排名，并绘制预测值对真实值的 case 可视化。 |

## 工具

| 文档 | 说明 |
|------|------|
| [inspect-config.md](inspect-config.md) | 使用 `tool/inspect_config.py` 预览配置的 sweep 展开（运行数、数据集、模型）。 |
| [aggregate-results.md](aggregate-results.md) | 使用 `tool/aggregate_results.py` 将某数据集的 `performance.csv` 和 `profile.csv` 合并为单一 CSV。 |
| [plot-bubble.md](plot-bubble.md) | 使用 `tool/plot_bubble.py` 从聚合 CSV 生成气泡图。 |
| [rank-models.md](rank-models.md) | 使用 `tool/rank_models.py` 按 `pred_len`/seed 对模型排名。 |
| [visualize-data.md](visualize-data.md) | 使用 `tool/visual_data.py` 从 TOML 配置绘制数据集样本。 |
| [dataset-characteristics.md](dataset-characteristics.md) | 使用 `tool/dataset_characteristics.py` 提取 TFB 风格的数据集特征（趋势/季节性/平稳性等）。 |
| [gift-eval.md](gift-eval.md) | 使用 `tool/gift_eval_download.py` 下载 GIFT-EVAL 数据集并运行 53 数据集 sweep。 |
| [scripts.md](scripts.md) | 统一 `tsf` 工具（脚手架 / smoke / run / aggregate-plot）+ `detect_hardware.sh`。 |
