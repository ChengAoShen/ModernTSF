---
model: "ASTGCN"
category: "covariate_prediction"
category_name: "协变量预测"
forecasting_setting: "covariate"
config: "configs/models/ASTGCN.toml"
registry: "models.astgcn.registry"
---
# ASTGCN

这是一个协变量预测模型，对应原空气质量预测设定。它面向节点目标值预测，并利用历史协变量以及部分模型支持的已知未来协变量。

简要说明：基于注意力的时空 GCN（在 Chebyshev 图卷积上叠加空间 + 时间注意力）。

在 ModernTSF 中，`ASTGCN` 的默认配置位于 `configs/models/ASTGCN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
