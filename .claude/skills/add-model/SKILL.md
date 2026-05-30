---
name: add-model
description: Guide the user through adding a new model to the ModernTSF project. Use when the user wants to integrate a new PyTorch model, register a custom architecture, or wire a new forecaster into the benchmark pipeline.
---

## When to use / what to ask

Ask the user for:
- **Model name** (PascalCase, e.g. `MyModel`) — used as the registry key and TOML `name`
- **Module name** (snake_case, e.g. `my_model`) — used as the directory and import path
- **Required hyperparameters** and their types/defaults (e.g. `enc_in: int`, `hidden_size: int = 128`)

---

## Steps

### 1. Create the model package

```
src/models/<model_name>/
  model.py      # nn.Module implementation
  schema.py     # Pydantic ModelParameterConfig
  registry.py   # register() function
```

**`schema.py`**
```python
from pydantic import BaseModel

class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 128
```

**`model.py`**
```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, enc_in: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(enc_in, hidden_size)

    def forward(self, x, *args):   # accept & ignore unused temporal-mark args
        return self.proj(x)
```

**`registry.py`**
```python
from benchmark.registry import MODEL_REGISTRY
from models.<model_name>.model import Model
from models.<model_name>.schema import ModelParameterConfig

def register() -> None:
    MODEL_REGISTRY.register(
        "<ModelName>",
        lambda cfg, params: Model(
            enc_in=params["enc_in"],
            hidden_size=params.get("hidden_size", 128),
        ),
        ModelParameterConfig,
    )
```

Factory signature: `lambda cfg, params: model_instance` where `cfg` is the validated `RootConfig`.

### 2. Register in MODEL_NAME_MAP

Edit `src/benchmark/registry/models.py`:
```python
MODEL_NAME_MAP["<ModelName>"] = "models.<model_name>.registry"
```

### 3. Create the model config

`configs/models/<ModelName>.toml`:
```toml
[model]
name = "<ModelName>"

[model.params]
enc_in = 7
hidden_size = 128
```

### 4. Use in a run config

```toml
extends = ["../../base.toml", "../../datasets/etth1.toml", "../../models/<ModelName>.toml"]
```

### 5. Run the experiment

```bash
uv run modern-tsf --config configs/runs/<your_run_config>.toml
```

---

## Key rules

- `forward` signature is `(self, x, x_mark, dec_inp, dec_mark)` — use `*args` to accept and ignore unused temporal marks.
- Registration is idempotent; calling `register()` twice is safe.
- `cfg` in the factory is the full `RootConfig`; use it to read top-level fields like `cfg.pred_len` if needed.

---

See `docs/en/add-model.md` for complete annotated examples.
