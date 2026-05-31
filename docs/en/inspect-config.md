# Inspect config

Preview sweep expansion for a run TOML without executing any training. Prints the total number of expanded runs and the unique values of datasets, models, prediction horizons, seeds, and every swept parameter.

## Usage

```bash
uv run python tool/inspect_config.py --config configs/runs/multi_sweep.toml
```

## Example output

```
Total runs: 32
Datasets: ETTh1, ETTm1
Models: DLinear, Linear
Pred lens: 96, 192, 336, 720
Seeds: 0, 1
Sweep values:
  extend.datasets: etth1, ettm1
  extend.models: DLinear, Linear
  experiment.random_seed: 0, 1
  task.pred_len: 96, 192, 336, 720
```

`Total runs` is the product of all sweep axes (here 2 datasets × 2 models × 4 pred\_lens × 2 seeds = 32). The `Sweep values` block lists each swept key and its distinct values; it is omitted when the config has no `[sweep]` section.

## Arguments

- `--config`: path to a run TOML file (required).

## Notes

- Reads and fully expands the config (including `extends` chains, `[sweep]`, and `[sweep.extend]`) but does not load any data or build any model.
- Useful for sanity-checking a sweep before a long run.
