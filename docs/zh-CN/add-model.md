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
