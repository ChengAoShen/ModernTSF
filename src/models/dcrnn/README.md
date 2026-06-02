---
model: "DCRNN"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/DCRNN.toml"
registry: "models.dcrnn.registry"
---
# DCRNN

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：扩散卷积循环网络（GRU 内做双向随机游走图卷积）。

在 ModernTSF 中，`DCRNN` 的默认配置位于 `configs/models/DCRNN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
