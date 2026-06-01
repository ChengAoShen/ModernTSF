---
model: "STPGNN"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/STPGNN.toml"
registry: "models.stpgnn.registry"
---
# STPGNN

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：时空关键节点图神经网络。

在 ModernTSF 中，`STPGNN` 的默认配置位于 `configs/models/STPGNN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
