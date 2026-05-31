---
name: add-dataset
description: Guide the user through adding a new dataset to the ModernTSF project. Use when the user wants to register a new dataset, integrate a CSV file or pre-split folder, or wire up a custom data source for benchmarking.
---

## When to use / what to ask

First ask the user:

1. **What is the dataset name?** (used for the config key and file names)
2. **What pattern fits your data?**
   - **Pattern C (custom CSV) — preferred for a plain CSV.** One CSV (a `date`
     column + numeric channels) that ModernTSF splits automatically. Uses the
     built-in `name = "custom"` loader — **no custom code at all**, just a config.
   - **Pattern B (pre-split)** — you already have `train.csv`, `val.csv`, `test.csv` in one folder. No custom code needed.
   - **Pattern A (single-file)** — one CSV needing a bespoke `_read_data` (unusual layout, synthetic generation). Requires new source files.
   - **Traffic / spatiotemporal bundle** — value array + adjacency matrix (PEMS, METR-LA). Build a node bundle with `tool/convert_traffic.py` (see bottom).

---

## Pattern C: custom CSV (no custom code) — preferred

For a standard CSV (a `date` column plus one column per channel), reuse the
built-in `custom` loader. Just create `configs/datasets/<name>.toml`:

```toml
[dataset]
name = "custom"
root_path = "./dataset/<folder>"
data_path = "<file>.csv"

[dataset.params]
target = "OT"
scale = true
split_ratio = [0.7, 0.1, 0.2]
```

This is how `exchange`, `ili`, `beijing_air`, `weather`, `electricity`, `traffic`
(and others) are wired. Use it in a run config via `extends`.

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

## Traffic / spatiotemporal bundle (PEMS, METR-LA)

For graph datasets with a value array + adjacency matrix, build a node bundle
with `tool/convert_traffic.py`, then point a `cauair_st` config at it:

```bash
uv run python tool/convert_traffic.py \
    --values dataset/metr_la/metr_la.npz --values-key data \
    --adj dataset/metr_la/adj_mx.pkl \
    --output-dir dataset/metr_la \
    --seq-len 12 --pred-len 12 --add-time --freq-min 5 \
    --splits 0.7,0.1,0.2
```

Then `configs/datasets/<name>.toml`:

```toml
[dataset]
name = "cauair_st"
root_path = "./dataset/metr_la"
data_path = ""

[dataset.params]
input_dim = 3       # value + time_in_day + day_in_week
npz_name = "his.npz"
scale = false
# adj_norm = "..."  # optional: normalize the adjacency for graph models
```

Set `[dataset.params] adj_norm` (run-time hook) to normalize the adjacency for
graph/spatiotemporal models. Existing bundles: `metr_la`, `pems_bay`, `pems03/04/07/08`.

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
