---
model: "LSTM"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/LSTM.toml"
registry: "models.lstm.registry"
---
# LSTM

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：逐节点的普通 LSTM 序列预测器。

在 ModernTSF 中，`LSTM` 的默认配置位于 `configs/models/LSTM.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
