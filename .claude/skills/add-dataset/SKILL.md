---
name: add-dataset
description: Guide the user through adding a new dataset to the ModernTSF project. Use when the user wants to register a new dataset, integrate a CSV file or pre-split folder, or wire up a custom data source for benchmarking.
---

## When to use / what to ask

First ask the user:

1. **What is the dataset name?** (used for the config key and file names)
2. **What pattern fits your data?**
   - **Pattern B (pre-split)** — you already have `train.csv`, `val.csv`, `test.csv` in one folder. No custom code needed.
   - **Pattern A (single-file)** — one CSV that ModernTSF should split automatically. Requires new source files.

---

## Pattern B: pre-split dataset (no custom code)

Folder layout required:

```
dataset/<name>/
  train.csv
  val.csv
  test.csv
```

All three files must share the same column layout. A `date` column is optional (zero timestamps used if absent). The scaler is always fitted on `train.csv`.

Create `configs/datasets/<name>.toml`:

```toml
[dataset]
name = "presplit"
root_path = "./dataset/<name>"
data_path = ""

[dataset.params]
target = "<target_column>"
scale = true
```

Use in a run config:

```toml
extends = ["../../base.toml", "../../datasets/<name>.toml", "../../models/DLinear.toml"]
```

---

## Pattern A: single-file dataset (split at load time)

### Step 1 — Create dataset implementation

`src/data/datasets/<name>.py` — inherit `ForecastingDataset`, implement `_read_data`:

```python
class Dataset_MyDataset(ForecastingDataset):
    def _read_data(self, flag, features, target, split_ratio, scale):
        df_raw = pd.read_csv(self.file_path)
        num_samples = len(df_raw)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)
        # feature selection and scaling ...
        return series_data, time_stamp
```

Add `register()` at the bottom of the same file:

```python
from benchmark.registry import DATASET_REGISTRY
from data.schemas.datasets.<name> import DatasetParameterConfig

def register() -> None:
    DATASET_REGISTRY.register("<name>", Dataset_MyDataset, DatasetParameterConfig)
```

### Step 2 — Define a parameter schema

`src/data/schemas/datasets/<name>.py`:

```python
from pydantic import BaseModel, Field

class DatasetParameterConfig(BaseModel):
    target: str
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])
```

### Step 3 — Register in DATASET_NAME_MAP

Edit `src/benchmark/registry/datasets.py`:

```python
DATASET_NAME_MAP["<name>"] = "data.datasets.<name>"
```

### Step 4 — Create dataset config

`configs/datasets/<name>.toml`:

```toml
[dataset]
name = "<name>"
root_path = "./dataset/<name>"
data_path = "<file>.csv"

[dataset.params]
target = "OT"
scale = true
split_ratio = [0.7, 0.1, 0.2]
```

### Step 5 — Use in a run config

```toml
extends = ["../../base.toml", "../../datasets/<name>.toml", "../../models/DLinear.toml"]
```

---

## Verify / run after adding

```bash
uv run modern-tsf --config configs/runs/run_single_data.toml
```

Or with the helper script (set DATASET to your new name):

```bash
DATASET=<name> bash scripts/aggregate_and_plot.sh
```

---

## Notes

- CSV datasets should include a `date` column for time feature generation (Pattern A requires it; Pattern B treats it as optional).
- For synthetic datasets (Pattern A), ignore `data_path` and generate series directly in `_read_data`.
- For single-target mode (`features = "S"`), use `target` to select the channel.
- After the dataset is set up, offer to create a run config that uses it.

Reference: [docs/en/add-dataset.md](../../../docs/en/add-dataset.md)
