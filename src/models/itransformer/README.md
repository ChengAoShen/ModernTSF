---
model: "iTransformer"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/iTransformer.toml"
registry: "models.itransformer.registry"
---
# iTransformer

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：倒置 Transformer：对通道做注意力，对时间做 FFN。

在 ModernTSF 中，`iTransformer` 的默认配置位于 `configs/models/iTransformer.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
