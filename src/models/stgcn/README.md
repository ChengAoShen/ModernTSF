---
model: "STGCN"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/STGCN.toml"
registry: "models.stgcn.registry"
---
# STGCN

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：时空图卷积网络（图卷积 + 时间卷积块）。

在 ModernTSF 中，`STGCN` 的默认配置位于 `configs/models/STGCN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
