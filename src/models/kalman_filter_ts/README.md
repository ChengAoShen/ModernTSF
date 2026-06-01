---
model: "KalmanFilterTS"
category: "time_series"
category_name: "时间序列"
forecasting_setting: "time_series"
config: "configs/models/KalmanFilterTS.toml"
registry: "models.kalman_filter_ts.registry"
---
# KalmanFilterTS

这是一个时间序列预测模型，面向普通单变量或多变量序列预测。它接收历史窗口中的数值序列，并输出未来预测窗口。

简要说明：Kalman Filter 启发的 alpha-beta 平滑器，更新增益可学习。

ModernTSF 当前注册的是 PyTorch 原生适配器，统一使用标准训练器和 `torch.nn.Module` 接口；当运行设备设为 CUDA/MPS 时，这些线性、核、树集成、统计和循环网络风格模型可以随张量迁移到加速设备。

在 ModernTSF 中，`KalmanFilterTS` 的默认配置位于 `configs/models/KalmanFilterTS.toml`，参数 schema 位于 `schema.py`，模型实现或适配器位于 `model.py`，注册入口位于 `registry.py`。
