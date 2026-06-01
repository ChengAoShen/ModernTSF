---
model: "DeepAir"
category: "covariate_prediction"
category_name: "协变量预测"
forecasting_setting: "covariate"
config: "configs/models/DeepAir.toml"
registry: "models.deepair.registry"
---
# DeepAir

这是一个协变量预测模型，对应原空气质量预测设定。它面向节点目标值预测，并利用历史协变量以及部分模型支持的已知未来协变量。

简要说明：基于融合的深度空气质量预测器。

在 ModernTSF 中，`DeepAir` 的默认配置位于 `configs/models/DeepAir.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
