---
model: "BigST"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/BigST.toml"
registry: "models.bigst.registry"
---
# BigST

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：线性复杂度时空 GNN，通过随机特征线性注意力扩展到大规模图。

在 ModernTSF 中，`BigST` 的默认配置位于 `configs/models/BigST.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
