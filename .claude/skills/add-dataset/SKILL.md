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

## The fast path (scaffold)

For the common patterns, one command generates the config (and, for `single`, the
loader + schema + `DATASET_NAME_MAP` entry):

```bash
# Pattern C — custom CSV (config only)
uv run python tool/tsf.py new-dataset --name my_csv --pattern custom \
    --root-path ./dataset/my_csv --data-path my_csv.csv --target OT

# Pattern B — pre-split train/val/test (config only)
uv run python tool/tsf.py new-dataset --name my_split --pattern presplit \
    --root-path ./dataset/my_split --target OT

# Pattern A — single-file with a bespoke loader (code + schema + wiring)
uv run python tool/tsf.py new-dataset --name my_special --pattern single \
    --root-path ./dataset/my_special --data-path my_special.csv --target OT
```

Then put the data under the `--root-path` folder and reference the config from a
run config via `extends`. For `single`, fill the real loader into
`src/data/datasets/<name>.py` `_read_data`. The pattern details below explain each
config's fields.

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

Only when an unusual layout or synthetic generation needs bespoke code. Five
files to wire (the full annotated code templates live in the doc — link below):

1. `src/data/datasets/<name>.py` — subclass `ForecastingDataset`, implement
   `_read_data` (use `self._get_borders` for the split), add a `register()` at
   the bottom that calls `DATASET_REGISTRY.register("<name>", cls, schema)`.
2. `src/data/schemas/datasets/<name>.py` — a Pydantic `DatasetParameterConfig`
   (`target`, `scale`, `split_ratio`, …).
3. `src/benchmark/registry/datasets.py` — add
   `DATASET_NAME_MAP["<name>"] = "data.datasets.<name>"`.
4. `configs/datasets/<name>.toml` — `name = "<name>"` + `[dataset.params]`.
5. Reference it from a run config via `extends`.

See [docs/en/add-dataset.md](../../../docs/en/add-dataset.md) Pattern A for the
complete copy-paste code for each step.

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

Then aggregate + plot the dataset's results in one shot:

```bash
uv run python tool/tsf.py aggregate-plot --dataset <name>
```

---

## Notes

- CSV datasets should include a `date` column for time feature generation (Pattern A requires it; Pattern B treats it as optional).
- For synthetic datasets (Pattern A), ignore `data_path` and generate series directly in `_read_data`.
- For single-target mode (`features = "S"`), use `target` to select the channel.
- After the dataset is set up, offer to create a run config that uses it.

Reference: [docs/en/add-dataset.md](../../../docs/en/add-dataset.md)
