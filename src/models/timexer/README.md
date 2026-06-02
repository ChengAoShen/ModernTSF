---
model: "TimeXer"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/TimeXer.toml"
registry: "models.timexer.registry"
---
# TimeXer

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：内生变量分块嵌入 + 外生变量倒置嵌入，通过全局 token 做交叉注意力。

在 ModernTSF 中，`TimeXer` 的默认配置位于 `configs/models/TimeXer.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
