Guide the user through adding a new dataset to the ModernTSF project.

First ask:
1. Does the data come as a **single CSV file** (to be split automatically) or as **pre-split files** (`train.csv`, `val.csv`, `test.csv` in one folder)?

## If pre-split (Pattern B)

No custom code needed. Just create a config file at `configs/datasets/<name>.toml`:

```toml
[dataset]
name = "presplit"
root_path = "./dataset/<folder_name>"
data_path = ""

[dataset.params]
target = "<target_column>"
scale = true
```

The folder must contain `train.csv`, `val.csv`, and `test.csv` with identical column layouts. A `date` column is optional.

## If single-file (Pattern A)

Walk through these steps:

1. **Create** `src/data/datasets/<name>.py` — inherit `ForecastingDataset`, implement `_read_data`.
2. **Create** `src/data/schemas/datasets/<name>.py` — Pydantic schema for `dataset.params`.
3. **Register** — add `register()` function in the dataset module and add the name to `DATASET_NAME_MAP` in `src/benchmark/registry/datasets.py`.
4. **Create** `configs/datasets/<name>.toml` — dataset config with `root_path`, `data_path`, and `params`.

Refer to `docs/en/add-dataset.md` for full code examples.

After the dataset is set up, offer to create a run config that uses it.
