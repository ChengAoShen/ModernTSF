# Add a new model

Models are registered through `MODEL_NAME_MAP` and a module-level `register()` function. Each model has a schema that validates its `model.params`.

## 1) Create the model package

Add a new module under `src/models/<model_name>/` with three files:

```text
src/models/my_model/
  model.py
  schema.py
  registry.py
```

## 2) Define the schema

`schema.py` should declare a `ModelParameterConfig` with fields used by the model.

```python
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 128
```

## 3) Implement the model

`model.py` provides the actual `torch.nn.Module`.

```python
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, enc_in: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(enc_in, hidden_size)

    def forward(self, x, *args):
        return self.proj(x)
```

## 4) Register the model

In `registry.py`, define `register()` and hook it into `MODEL_REGISTRY`.

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

The factory signature is `lambda cfg, params: model`, where `cfg` is the validated root config.

## 5) Add to MODEL_NAME_MAP

Edit `src/benchmark/registry/models.py`:

```python
MODEL_NAME_MAP["MyModel"] = "models.my_model.registry"
```

## 6) Add a model config

Create `configs/models/MyModel.toml`:

```toml
[model]
name = "MyModel"

[model.params]
enc_in = 7
hidden_size = 128
```

## 7) Use in a run config

Create or update a run config to include the model:

```toml
extends = ["../../base.toml", "../../datasets/etth1.toml", "../../models/MyModel.toml"]
```

You can now run the experiment with `modern-tsf`.
