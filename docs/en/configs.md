# Config loading and usage

ModernTSF is TOML-first. Configs are composed using `extends`, validated by Pydantic schemas, and optionally expanded by `[sweep]` into multiple runs.

## Config layout

Typical layout:

- `configs/base.toml`: shared defaults for all runs.
- `configs/datasets/*.toml`: dataset-specific config.
- `configs/models/*.toml`: model-specific config.
- `configs/runs/*.toml`: entry-point configs that combine the above with `extends`.

Example run config (`configs/runs/run_single_data.toml`):

```toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[evaluation]
enable_profile = true

[sweep.task]
pred_len = [96, 192, 336, 720]
```

## How `extends` works

`extends` can be a string or a list of TOML paths. The loader resolves each path relative to the config file and merges them in order.

- Dicts are deep-merged.
- Scalars and lists are replaced by later values.
- The final config is validated by `RootConfig` in `src/benchmark/config/schema/root.py`.

## Sweep expansion

`[sweep]` expands to a cartesian product of values.

Supported styles:

```toml
[sweep]
experiment.random_seed = [0, 1, 2]

[sweep.task]
pred_len = [96, 192, 336, 720]
```

Each combination becomes an independent run. The CLI executes them sequentially.

## Sweep extend (multi-config axes)

You can expand runs by referencing multiple TOML files directly from `sweep.extend`.

```toml
extends = ["../../base.toml", "../../models/DLinear.toml"]

[sweep.extend]
datasets = [
  "../../datasets/electricity.toml",
  "../../datasets/etth1.toml",
]
models = [
  "../../models/DLinear.toml",
  "../../models/Linear.toml",
]

[sweep.task]
pred_len = [96, 192]
```

- Each `sweep.extend.<axis>` is an arbitrary axis name.
- Paths are resolved relative to the run config file.
- The loader computes a cartesian product over all extend axes.
- Merge order: `extends` < `sweep.extend` configs < current run config.
- Sweep metadata is recorded as `sweep.extend.<axis>` in the CSV summary output.

## Multi-sweep strategy

`sweep.extend` expands first, then the remaining `[sweep]` keys are expanded. The final total runs are:

```text
total = product(len(values) for each sweep.extend axis)
        * product(len(values) for each sweep key)
```

Example (`configs/runs/multi_sweep.toml`):

- 2 datasets × 2 models from `sweep.extend`
- 2 random seeds × 2 pred_len values from `sweep`
- Total runs = 2 × 2 × 2 × 2 = 16

## Inspecting a config

Use the helper script to preview how a config expands:

```bash
uv run python tool/inspect_config.py --config configs/runs/multi_sweep.toml
```

It reports total runs, covered datasets/models, and sweep value ranges.

## Running a config

Use the CLI entry point:

```bash
uv run modern-tsf --config configs/runs/run_single_data.toml
```

Outputs are written to `experiment.work_dir`, with per-run subdirectories for checkpoints and CSV summaries.

Each run logs a summary line before training, including model, dataset, key task settings, and sweep values when present.

## Dataset-only configs

The visualization tool can load dataset-only configs (e.g. `configs/datasets/periodic.toml`) and fills missing task values with defaults from `configs/base.toml`.
