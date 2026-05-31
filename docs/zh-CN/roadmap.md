# 范围

ModernTSF 是一个**纯预测（forecasting-only）**基准。它面向三种数据设定下的预测（`task.mode`，见 [task-modes.md](task-modes.md)）：

- `time_series`——经典多变量预测 `(B, T, C)`。
- `spatiotemporal`——带邻接矩阵的节点结构化预测。
- `covariate`——时空 + 未来已知协变量。

全部 99 个模型，以及所有数据集、指标、损失、评估路径，都服务于这三种预测设定。`task.mode` 只暴露上述三种设定，因此**所有可达代码路径都是预测**。部分上游 TSLib 风格模型（Autoformer、FEDformer、TimesNet、TiDE、SegRNN、CrossLinear、MoFo）原本携带的多任务 `task_name` 分支已在移植时移除——代码库中不存在任何 `task_name` 模型参数、schema 字段或非预测分支。

## 明确越界（不计划）

以下是**不同的任务类型**、不属于预测，因此有意**不**纳入 ModernTSF：**插补（imputation）**、**异常检测**、**分类**、**基础模型预训练**（零样本 LLM 预测器 / 大规模预训练语料）。它们各自需要专属的数据格式、任务契约与评估协议，加入它们超出本项目范围。

## 已从 benchmark 调研采纳（完成）

从 BasicTS / TSLib / TFB 采纳的非模型资产（均服务于上述三种预测设定）：额外指标（`corr`/`rse`/`wape`/`smape`，`mase` 选开）、masked 损失（用于缺失值预测）、邻接归一化工具 + `adj_norm`、大量 CSV + 交通数据集、可插拔训练 callback 层（课程学习 / 梯度裁剪 / 梯度累积 / 辅助损失）、scaler 增强、fit/inference 计时、聚合公平性（`--null-threshold`）、RollingForecast 评估策略、数据集特征提取器。
