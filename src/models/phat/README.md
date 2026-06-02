---
model: "PHAT"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/PHAT.toml"
registry: "models.phat.registry"
---
# PHAT

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：周期异质性 Transformer；`PHAT_Attention` ⚠️ **未验证**的论文重建（arXiv:2602.00654），非论文复现。

在 ModernTSF 中，`PHAT` 的默认配置位于 `configs/models/PHAT.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
