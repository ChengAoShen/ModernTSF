# 路线图——延后 / 越界的任务

ModernTSF 目前面向**预测（forecasting）**的三种数据设定（`time_series` / `spatiotemporal` / `covariate`，见 [task-modes.md](task-modes.md)）。下列条目是净新的**任务类型**（不是三种预测模式的增强），有意延后：每个都需要自己的数据格式和评估协议，无法在当前预测框架内做 smoke 验证。部分支撑构件已就位，逐项列出。

## M4 短期预测赛道（延后）

M4 赛道（单变量序列集合、按频率季节性、无输入归一化、用 SMAPE/MASE/OWA 对 Naive2 基线打分）是与长期多变量预测不同的子范式。

- **需要**：M4 数据集类（M4 发布的按频率 Train/Test CSV）、无 scaler 单变量路径、Naive2 基线、OWA 聚合。
- **已就位**：`smape` 与 `mase` 指标（`METRIC_NAME_MAP`）；滚动评估器可承载逐序列打分。
- **延后原因**：需要 M4 数据集下载，以及与窗口化多变量契约不同的单变量集合加载器+协议。

## 独立插补任务模式（延后）

掩码插补（遮住观测时间步、重建、仅在被遮位置打分）与预测是不同任务。

- **已就位**：masked 损失 `masked_mae` / `masked_mse` / `masked_rmse`（带 `targets_mask`，BasicTS 约定）——缺失值/插补训练的损失侧构件。
- **需要**：`task.mode = "imputation"` 数据路径（随机输入掩码、无未来窗口）、插补评估路径、按掩码比例报告。
- **延后原因**：改变任务语义与四元组数据契约。

## 其他越界任务（不计划）

与模型移植的范围一致，以下保持越界（属不同任务，非预测）：**异常检测**（PSM/MSL/SMAP/SMD/SWaT）、**分类**（UEA）、**基础模型预训练**（如 BLAST 式语料、需亿级 checkpoint 的零样本 LLM 预测器）。

## 已从 benchmark 调研采纳（完成）

参考：以下来自 BasicTS / TSLib / TFB 的非模型资产已采纳（见相关文档）：额外指标（`corr`/`rse`/`wape`/`smape`，`mase` 选开）、masked 损失、邻接归一化工具 + `adj_norm`、大量 CSV + 交通数据集、可插拔训练 callback 层（课程学习 / 梯度裁剪 / 梯度累积 / 辅助损失）、scaler 增强、fit/inference 计时、聚合公平性（`--null-threshold`）、RollingForecast 评估策略、数据集特征提取器。
