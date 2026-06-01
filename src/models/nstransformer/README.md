---
model: "NSTransformer"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/NSTransformer.toml"
registry: "models.nstransformer.registry"
---
# NSTransformer

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：非平稳 Transformer，结合去平稳注意力与序列平稳化。

在 ModernTSF 中，`NSTransformer` 的默认配置位于 `configs/models/NSTransformer.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
