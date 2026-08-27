---
name: smoke-models
description: Run fast end-to-end smoke checks for one or more ModernTSF models. Use after implementation changes or when validating training and output shape quickly; not for exhaustive repository or paper audits.
---

# Smoke-test models

```bash
uv run tsf smoke --model DLinear
uv run tsf smoke --config configs/runs/smoke_dlinear.toml
uv run tsf smoke --all --jobs 8
```

Choose the narrowest scope. Each case should exercise construction, a short training path, evaluation, and declared output shape. Add focused cases for material optional objectives or output types. Success means every selected config reports PASS; preserve the final diagnostic for each failure.
