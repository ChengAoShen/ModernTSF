---
model: "OLinear"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/OLinear.toml"
registry: "models.olinear.registry"
venue: "NeurIPS 2025"
upstream: "https://github.com/jackyue1994/OLinear"
---
# OLinear

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：正交变换线性预测适配器，带归一化通道混合。

ModernTSF 当前注册的是轻量原生适配器，统一使用 `src/models/_recent_tsf.py` 的预测接口与归一化路径；它记录并参考公开仓库的核心建模偏置，但不直接复制上游训练工程。

在 ModernTSF 中，`OLinear` 的默认配置位于 `configs/models/OLinear.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
