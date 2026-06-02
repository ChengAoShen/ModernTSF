---
model: "AirPhyNet"
category: "covariate_prediction"
category_name: "协变量预测"
forecasting_setting: "covariate"
config: "configs/models/AirPhyNet.toml"
registry: "models.airphynet.registry"
---
# AirPhyNet

这是一个协变量预测模型，对应原空气质量预测设定。它面向节点目标值预测，并利用历史协变量以及部分模型支持的已知未来协变量。

简要说明：物理信息网络，基于扩散/平流 ODE（需 `torchdiffeq`）。

在 ModernTSF 中，`AirPhyNet` 的默认配置位于 `configs/models/AirPhyNet.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
