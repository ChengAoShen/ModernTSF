# 如何加入新模型

模型通过 `MODEL_NAME_MAP` 与模块级 `register()` 注册。每个模型都有 schema 用于校验 `model.params`。

## 1) 创建模型目录

在 `src/models/<model_name>/` 下新增：

```text
src/models/my_model/
  model.py
  schema.py
  registry.py
```

## 2) 编写参数 schema

`schema.py` 定义 `ModelParameterConfig`：

```python
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 128
```

## 3) 实现模型

`model.py` 中实现 `torch.nn.Module`：

```python
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, enc_in: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(enc_in, hidden_size)

    def forward(self, x, *args):
        return self.proj(x)
```

## 4) 注册模型

`registry.py` 中添加 `register()`：

```python
from benchmark.registry import MODEL_REGISTRY
from models.my_model.model import Model
from models.my_model.schema import ModelParameterConfig


def register() -> None:
    MODEL_REGISTRY.register(
        "MyModel",
        lambda cfg, params: Model(
            enc_in=params["enc_in"],
            hidden_size=params.get("hidden_size", 128),
        ),
        ModelParameterConfig,
    )
```

工厂函数签名为 `lambda cfg, params: model`，其中 `cfg` 为完整配置对象。

## 5) 更新 MODEL_NAME_MAP

编辑 `src/benchmark/registry/models.py`：

```python
MODEL_NAME_MAP["MyModel"] = "models.my_model.registry"
```

## 6) 添加模型配置

新建 `configs/models/MyModel.toml`：

```toml
[model]
name = "MyModel"

[model.params]
enc_in = 7
hidden_size = 128
```

## 7) 在入口配置中使用

```toml
extends = ["../../base.toml", "../../datasets/etth1.toml", "../../models/MyModel.toml"]
```

然后使用 `modern-tsf` 运行即可。

## 时空 / 空气质量模型

当 `task.mode = "spatiotemporal"` 或 `"covariate"` 时，模型的 `forward` 收到数值张量 `x_enc`，形状 `(B, T, N)`，以及**节点结构化**的协变量标记 `x_mark_enc`，形状 `(B, T, N, F)`（covariate 还会有未来协变量块 `x_mark_dec`，形状 `(B, pred_len, N, F)`）。用 `src/models/_external/marks.py` 里的共享辅助函数构造 `(B, T, N, 1 + F)` 输入：

```python
from models._external.marks import to_spatiotemporal, future_time_features

st_input = to_spatiotemporal(x_enc, x_mark_enc)        # (B, T, N, 1 + F)
future = future_time_features(x_mark_dec, n=x_enc.shape[-1])  # (B, T, N, F)
```

这些辅助函数是多态的：3 维 `(B, T, 6)` 标记被当作原始日历时间戳，4 维 `(B, T, N, F)` 标记被当作节点协变量，因此同一个适配器在 forecasting 与节点结构化模式下都能工作。批次形状见 `docs/zh-CN/task-modes.md`，可参考现有的 `BiST` / `CauAir` 适配器作为范例。

## 高级训练目标

大多数模型只需要返回预测张量，并使用配置中的损失函数训练。只有当目标无法用标准预测损失表达时，才使用下面的 opt-in 约定：

- 加性正则：在 `forward` 中把有限标量张量写到 `self.aux_loss`，trainer 会把它加到配置损失上。
- 替代训练目标：在 `forward` 中写 `self.train_loss_override`，trainer 会用这个标量替代配置损失；验证和早停仍使用观测空间损失。
- 需要未来目标的训练目标：声明 `requires_train_target = True`，并实现 `set_train_target(self, y: torch.Tensor | None)`。trainer 只会在训练 forward 前传入原始未来目标，并在验证/评估前清空。该 target 必须视为 one-shot 训练状态。
- 模型自带预训练：实现 `pretrain(self, train_loader, device)`。它会在 optimizer 构造前运行一次，耗时计入 `train_time_sec` / `fit_time`。

规则：每次 `forward` 开始都应清空 `train_loss_override`；验证/评估不能依赖训练 target；除非必须替代整个训练目标，否则优先使用 `aux_loss`。声明 `requires_train_target` 的模型不支持 `torch.nn.DataParallel`。
