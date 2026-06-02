---
model: "DGCRN"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/DGCRN.toml"
registry: "models.dgcrn.registry"
---
# DGCRN

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：动态图卷积循环网络（GRU 内使用随时间变化的邻接）。

在 ModernTSF 中，`DGCRN` 的默认配置位于 `configs/models/DGCRN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
