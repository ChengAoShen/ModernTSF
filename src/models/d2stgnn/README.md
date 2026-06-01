---
model: "D2STGNN"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/D2STGNN.toml"
registry: "models.d2stgnn.registry"
---
# D2STGNN

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：解耦动态时空图网络（用动态图分离扩散信号与固有信号）。

在 ModernTSF 中，`D2STGNN` 的默认配置位于 `configs/models/D2STGNN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
