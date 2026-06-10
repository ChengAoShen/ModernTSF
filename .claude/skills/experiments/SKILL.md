---
name: experiments
description: Run one-click ablation studies and hyperparameter sweeps via the [sweep] config mechanism, and visualize a trained model's forecasts vs ground truth. Use when the user wants to ablate model components, search hyperparameters, compare model variants across a grid, or plot prediction case studies.
---

## When to use

- **Ablation** — toggle a model's components on/off via `[sweep]` and compare.
- **Hyperparameter search** — vary numeric/architectural params over a grid.
- **Case visualization** — plot a trained model's forecasts against ground truth.

## Ablation / hyperparameter sweeps

Write a run config that `extends` a base + dataset + model, then sweep the params (a cartesian product expanded at load time):

```toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[sweep]
model.params.individual = [true, false]   # ablate a component on/off
model.params.kernel_size = [13, 25, 49]   # or sweep a hyperparameter

[sweep.task]
pred_len = [96, 336]
```

Swap whole model variants via `[sweep.extend]` (see `configs/runs/sweep_model.toml`); for datasets × models × seeds grids see `configs/runs/multi_sweep.toml`. Preview with the `inspect` skill, then run:

```bash
uv run modern-tsf --config configs/runs/<your_sweep>.toml
```

Set `[evaluation] enable_profile = true` to also record params/MACs. Afterwards use the `aggregate` / `rank` skills on the results.

## Case visualization (forecast vs truth)

Train first so a checkpoint exists, then:

```bash
uv run python tool/visualize_predictions.py \
    --config <same_run_config> --num-samples 4 --channel -1
```

Auto-finds the latest checkpoint for the `(dataset, model)`; `--checkpoint` pins a specific one. Works for spatiotemporal/covariate models too — pass a node index via `--channel`. Output: `work_dirs/<dataset>/<model>/cases.png`.

## Reference

Full guide: `docs/en/experiments.md`; sweep semantics: `docs/en/configs.md`.
