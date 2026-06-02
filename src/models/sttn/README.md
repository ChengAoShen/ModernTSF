---
model: "STTN"
category: "spatiotemporal_learning"
category_name: "时空学习"
forecasting_setting: "spatiotemporal"
config: "configs/models/STTN.toml"
registry: "models.sttn.registry"
---
# STTN

这是一个时空学习模型，面向节点结构化或图结构数据。它同时建模时间依赖与节点之间的空间关系，用于预测各节点未来目标值。

简要说明：时空 Transformer 网络（解耦的空间 + 时间注意力）。

在 ModernTSF 中，`STTN` 的默认配置位于 `configs/models/STTN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
