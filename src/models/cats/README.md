---
model: "CATS"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/CATS.toml"
registry: "models.cats.registry"
---
# CATS

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：查询自适应掩码 Transformer，对未来 token 做交叉注意力。

在 ModernTSF 中，`CATS` 的默认配置位于 `configs/models/CATS.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
