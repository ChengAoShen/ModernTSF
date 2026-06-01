---
model: "BiMamba"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/BiMamba.toml"
registry: "models.bimamba.registry"
---
# BiMamba

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：双向 Mamba，对序列正向与反向各扫描一次；无需 CUDA 算子的选择性扫描。

在 ModernTSF 中，`BiMamba` 的默认配置位于 `configs/models/BiMamba.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
