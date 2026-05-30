---
name: inspect
description: Preview sweep expansion for a run config — reports total run count, datasets, models, pred lengths, seeds, and per-axis sweep values without launching any training. Use when the user wants to preview how many runs or which datasets/models a config expands to before launching an experiment.
---

## When to use

Invoke when the user asks how many runs a config produces, which datasets/models are covered, or what values a sweep axis takes — before committing to a full experiment run.

If the user does not provide a config path, ask:
- Which run config file to inspect? (e.g. `configs/runs/run_single_data.toml`)

## Command

```bash
uv run python tool/inspect_config.py --config <run_config>
```

| Placeholder | Description |
|---|---|
| `<run_config>` | Path to a run TOML file (required) |

**Only flag:** `--config` (required). There are no other CLI flags.

## Output

Prints to stdout:

```
Total runs: <N>
Datasets: <comma-separated>
Models: <comma-separated>
Pred lens: <comma-separated>
Seeds: <comma-separated>
Sweep values:
  <axis>: <values>   # only shown when sweep axes exist
```

## Notes

- Works with any TOML that `load_config` can parse, including configs that use `extends`, `[sweep]`, and `[sweep.extend]`.
- Does not train, evaluate, or write any files — safe to run at any time.
- To actually run the experiment after inspecting, use `uv run modern-tsf --config <run_config>` or `bash scripts/run_multi_configs.sh <run_config>`.

## Reference

See `docs/en/configs.md` for full config syntax, `extends` chains, and sweep expansion rules.
