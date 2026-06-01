---
model: "GWNet"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/GWNet.toml"
registry: "models.gwnet.registry"
---
# GWNet

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：Graph WaveNet：自适应邻接 + 膨胀因果卷积。

在 ModernTSF 中，`GWNet` 的默认配置位于 `configs/models/GWNet.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
