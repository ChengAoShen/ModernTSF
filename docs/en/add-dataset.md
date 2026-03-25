# Add a new dataset

Datasets are registered through `DATASET_NAME_MAP` and a module-level `register()` function. Each dataset has a schema that validates `dataset.params`.

There are two patterns depending on whether your data comes as a single file (split at load time) or as pre-split files.

---

## Pattern A: single-file dataset (split at load time)

Use this pattern when you have one CSV file and want ModernTSF to split it into train/val/test automatically.

### 1) Create dataset implementation

Add a module under `src/data/datasets/`:

```text
src/data/datasets/my_dataset.py
```

Inherit `ForecastingDataset` and implement `_read_data`.

Key responsibilities in `_read_data`:

- Load raw data (CSV, parquet, synthetic, etc.).
- Apply feature selection (`features` in {`M`, `MS`, `S`}).
- Apply scaling if requested.
- Split by `split_ratio` using `_get_borders` (or custom logic for synthetic datasets).
- Return `(series_data, time_stamp)` as `np.ndarray`.

```python
class Dataset_Custom(ForecastingDataset):
    def _read_data(self, flag, features, target, split_ratio, scale):
        df_raw = pd.read_csv(self.file_path)
        num_samples = len(df_raw)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)
        # ... feature selection and scaling ...
        return series_data, time_stamp
```

### 2) Define a parameter schema

Create `src/data/schemas/datasets/my_dataset.py`:

```python
from pydantic import BaseModel, Field


class DatasetParameterConfig(BaseModel):
    target: str
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])
```

### 3) Register the dataset

In your dataset module, add a `register()` function:

```python
from benchmark.registry import DATASET_REGISTRY
from data.schemas.datasets.my_dataset import DatasetParameterConfig


def register() -> None:
    DATASET_REGISTRY.register("my_dataset", Dataset_My, DatasetParameterConfig)
```

### 4) Add to DATASET_NAME_MAP

Edit `src/benchmark/registry/datasets.py`:

```python
DATASET_NAME_MAP["my_dataset"] = "data.datasets.my_dataset"
```

### 5) Add a dataset config

Create `configs/datasets/my_dataset.toml`:

```toml
[dataset]
name = "my_dataset"
root_path = "./dataset/my_dataset"
data_path = "my.csv"

[dataset.params]
target = "OT"
scale = true
split_ratio = [0.7, 0.1, 0.2]
```

### 6) Use in a run config

```toml
extends = ["../../base.toml", "../../datasets/my_dataset.toml", "../../models/DLinear.toml"]
```

---

## Pattern B: pre-split dataset (train/val/test files in one folder)

Use this pattern when your data is already split into separate files. ModernTSF's built-in `presplit` dataset handles this without any custom code.

### Folder layout

```text
dataset/my_dataset/
  train.csv
  val.csv
  test.csv
```

All three files must share the same column layout. A `date` column is optional — time features are built from it if present, otherwise zero timestamps are used.

### Dataset config

```toml
[dataset]
name = "presplit"
root_path = "./dataset/my_dataset"
data_path = ""

[dataset.params]
target = "OT"
scale = true
```

The scaler is always fitted on `train.csv` so val/test receive consistent normalisation.

### Use in a run config

```toml
extends = ["../../base.toml", "../../datasets/my_dataset.toml", "../../models/DLinear.toml"]
```

No custom dataset class or schema is required.

---

## Notes

- CSV datasets should include a `date` column for time feature generation (Pattern A requires it; Pattern B treats it as optional).
- For synthetic datasets using Pattern A, you can ignore `data_path` and generate series directly in `_read_data`.
- For single-target mode (`features = "S"`), use `target` to select the channel.
