---
model: "MoFo"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/MoFo.toml"
registry: "models.mofo.registry"
---
# MoFo

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：周期模式 Transformer，周期对齐 patch。

在 ModernTSF 中，`MoFo` 的默认配置位于 `configs/models/MoFo.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
