---
name: add-dataset
description: Register a new dataset in ModernTSF from a standard CSV, custom loader, or traffic bundle. Use for dataset integration and configuration; not merely for inspecting or preprocessing an existing dataset.
---

# Add a dataset

Choose the smallest supported pattern: `custom` for one standard CSV, `single` only when a loader is required, or the traffic converter for values plus adjacency.

```bash
uv run tsf dataset add --name my_data --pattern custom \
  --root-path ./dataset/my_data --data-path my_data.csv --target OT
```

For traffic bundles, inspect `uv run tsf dataset convert-traffic --help` and provide explicit inputs, splits, and windows. Then inspect and exercise the dataset:

```bash
uv run tsf dataset inspect --config configs/datasets/my_data.toml
uv run tsf inspect --config configs/runs/<run>.toml
uv run tsf run configs/runs/<run>.toml
```

Confirm train-only scaling, feature/target selection, split boundaries, and adjacency injection. Stop before overwriting an existing dataset unless replacement was requested.

`tsf dataset add` also regenerates the preset's canonical README card. Complete
any loader-specific facts in executable config/code, regenerate with
`uv run python scripts/generate_resource_cards.py`, then require both
`uv run tsf dataset show my_data` and `uv run tsf dataset audit` to pass.
