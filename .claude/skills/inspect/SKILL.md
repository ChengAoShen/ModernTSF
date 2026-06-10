---
name: inspect
description: Preview sweep expansion for a run config — reports total run count, datasets, models, pred lengths, seeds, and per-axis sweep values without launching any training. Use when the user wants to preview how many runs or which datasets/models a config expands to before launching an experiment.
---

## When to use

The user asks how many runs a config produces, or which datasets/models/pred_lens a sweep covers — before committing to a full run. Ask for the config path if not given.

## Command

```bash
uv run python tool/inspect_config.py --config <run_config>
```

Prints total runs, datasets, models, pred lens, seeds, and per-axis sweep values. `--config` is the only flag. Read-only — trains nothing, writes nothing; works with any TOML that uses `extends`, `[sweep]`, or `[sweep.extend]`.

To actually run the experiment afterwards, use the `run` skill.

## Reference

Config syntax, `extends` chains, and sweep expansion rules: `docs/en/configs.md`.
