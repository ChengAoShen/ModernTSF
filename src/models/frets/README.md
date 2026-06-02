---
model: "FreTS"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/FreTS.toml"
registry: "models.frets.registry"
---
# FreTS

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：在频域实部/虚部分量上应用 MLP。

在 ModernTSF 中，`FreTS` 的默认配置位于 `configs/models/FreTS.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
