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

### Shortcut: `name = "custom"` for plain CSVs

If your data is a single flat-multivariate CSV (a `date` column plus numeric
channels), you do **not** need steps 1–4. Set `name = "custom"` in the config
and it reuses the built-in `Dataset_Custom` loader — config only, no new code:

```toml
[dataset]
name = "custom"
root_path = "./dataset/exchange_rate"
data_path = "exchange_rate.csv"

[dataset.params]
target = "OT"
scale = true
split_ratio = [0.7, 0.1, 0.2]
```

Shipped examples: `exchange`, `ili`, `beijing_air`, `aqshunyi`, `aqwan`, `nn5`,
`fred_md`.

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

---

## Pattern C: node-structured dataset (spatiotemporal / covariate)

Use this pattern for `task.mode = "spatiotemporal"` or `"covariate"`, where
each of `N` nodes carries a value plus `F` per-node covariates. Such a dataset
returns the standard four-item contract with the **value** in the series slots
and the **covariates** in the stamp slots:

```
__getitem__ -> (value_hist (T,N), value_fut (T,N), cov_hist (T,N,F), cov_fut (T,N,F))
```

`cov_fut` is the future covariate block consumed by air-quality models. A
node-structured dataset is a plain `torch.utils.data.Dataset` (it does not need
`ForecastingDataset`); set a class attribute `spatiotemporal = True` for clarity
and implement `__len__` / `__getitem__` / `inverse_transform`.

Two built-in examples:

- `synthetic_st` (`src/data/datasets/synthetic_st.py`) — generates a small
  `(T, N, 3)` tensor with calendar covariates `[time_in_day, day_in_week]`.
- `cauair_st` / `cauair_ts` (`src/data/datasets/cauair.py`) — load CauAir's
  index-windowed `.npz` bundles (`data (T, N, C)`, `idx_{train,val,test}.npy`,
  optional `adj_mx.npy`). `cauair_st` exposes the node layout for
  spatiotemporal / covariate modes; `cauair_ts` flattens the `N` node values
  into `C` channels for plain forecasting.

```toml
[dataset]
name = "cauair_st"
root_path = "./dataset/cauair_ccaq"
data_path = ""

[dataset.params]
input_dim = 8      # value + (input_dim - 1) covariates
npz_name = "his.npz"
scale = true
```

Traffic graph bundles (`metr_la`, `pems_bay`, `pems03/04/07/08`) reuse this same
`cauair_st` loader — there is no dedicated traffic dataset. Convert the raw value
matrix + adjacency into the bundle with `tool/convert_traffic.py`. See
`docs/en/datasets-traffic.md` for how to convert the raw arrays and which
configs to point at them.

When a graph model consumes the bundle's `adj_mx.npy`, you can have the runner
normalize that adjacency before injection by setting `adj_norm` under
`[dataset.params]` (e.g. `adj_norm = "gcn"`); the raw matrix is used unchanged
when it is unset. See the scheme table in `docs/en/task-modes.md`.

See `docs/en/task-modes.md` for how each mode shapes the batch and which models
are compatible.
