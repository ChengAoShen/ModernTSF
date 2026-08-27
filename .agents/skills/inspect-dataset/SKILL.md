---
name: inspect-dataset
description: Inspect, profile, or visualize an existing ModernTSF dataset. Use for resolved shapes, trend and seasonality characteristics, split checks, or raw sample plots; not for result plots or model predictions.
---

# Inspect a dataset

```bash
uv run tsf dataset inspect --config configs/datasets/<name>.toml --split train --per-channel
uv run tsf dataset plot --config configs/datasets/<name>.toml --split train --num-samples 3
```

Check split boundaries, tensor dimensions, missing values, target-channel behavior, leakage, inferred seasonal period, and adjacency metadata. Report generated artifact paths and distinguish observations from inferred characteristics.
