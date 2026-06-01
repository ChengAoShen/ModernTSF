---
model: "CrossGNN"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/CrossGNN.toml"
registry: "models.crossgnn.registry"
---
# CrossGNN

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：跨尺度、跨变量图网络，无需外部邻接矩阵即可建模多尺度交互。

在 ModernTSF 中，`CrossGNN` 的默认配置位于 `configs/models/CrossGNN.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
