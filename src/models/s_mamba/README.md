---
model: "S_Mamba"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/S_Mamba.toml"
registry: "models.s_mamba.registry"
---
# S_Mamba

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：iTransformer 风格的倒置嵌入，在通道维上叠加 Mamba 块；无需 CUDA 算子的选择性扫描。

在 ModernTSF 中，`S_Mamba` 的默认配置位于 `configs/models/S_Mamba.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
