---
model: "APN"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/APN.toml"
registry: "models.apn.registry"
venue: "AAAI 2026"
upstream: "https://github.com/decisionintelligence/APN"
---
# APN

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：自适应周期网络风格预测器，使用相位投影。

ModernTSF 当前注册的是轻量原生适配器，统一使用 `src/models/_recent_tsf.py` 的预测接口与归一化路径；它记录并参考公开仓库的核心建模偏置，但不直接复制上游训练工程。

在 ModernTSF 中，`APN` 的默认配置位于 `configs/models/APN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
