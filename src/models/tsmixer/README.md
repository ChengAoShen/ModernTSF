---
model: "TSMixer"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/TSMixer.toml"
registry: "models.tsmixer.registry"
---
# TSMixer

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：时间序列 MLP-Mixer，交替做时间与通道混合。

在 ModernTSF 中，`TSMixer` 的默认配置位于 `configs/models/TSMixer.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
