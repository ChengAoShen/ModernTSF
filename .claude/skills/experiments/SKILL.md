---
name: experiments
description: Run one-click ablation studies and hyperparameter sweeps via the [sweep] config mechanism, and visualize a trained model's forecasts vs ground truth. Use when the user wants to ablate model components, search hyperparameters, compare model variants across a grid, or plot prediction case studies.
---

## When to use

- **Ablation** — toggle a model's components on/off and compare.
- **Hyperparameter search** — vary numeric/architectural params over a grid.
- **Case visualization** — plot a trained model's forecast against the ground truth for a few test windows.

The first two are driven by `[sweep]` in a run config (a cartesian product expanded at load time). The third is `tool/visualize_predictions.py`. Full reference: `docs/en/experiments.md`.

## 1 & 2 — Ablation / hyperparameter sweeps

Write a run config that `extends` a base + dataset + model, then sweep the
`[model.params]` (ablation) or numeric/architectural params (hparam search):

```toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[sweep]
model.params.individual = [true, false]   # ablate a component on/off
model.params.kernel_size = [13, 25, 49]   # or sweep a hyperparameter

[sweep.task]
pred_len = [96, 336]
```

Swap whole model variants instead of fields via `[sweep.extend]` — see
`configs/runs/sweep_model.toml`. For datasets × models × seeds × pred_len grids,
see `configs/runs/multi_sweep.toml`.

Preview the grid (run count, datasets, models) before launching:

```bash
uv run python tool/inspect_config.py --config configs/runs/<your_sweep>.toml
```

Run it (single process, or sequentially on a GPU):

```bash
uv run modern-tsf --config configs/runs/<your_sweep>.toml
# or
uv run python tool/tsf.py run configs/runs/<your_sweep>.toml --gpus 0
```

Set `[evaluation] enable_profile = true` to also record params/MACs per run.
Aggregate/rank the resulting `performance.csv` rows with the `aggregate` / `rank` skills.

## 3 — Case visualization (forecast vs truth)

Train the model first so a checkpoint exists, then plot:

```bash
uv run modern-tsf --config configs/runs/run_single_data.toml

uv run python tool/visualize_predictions.py \
    --config configs/runs/run_single_data.toml \
    --num-samples 4 --channel -1
```

| Flag | Default | Description |
|---|---|---|
| `--config` | (required) | Run config TOML (same one used to train) |
| `--num-samples N` | — | Number of test windows to plot |
| `--channel I` | — | Channel/node index (`-1` = last) |
| `--checkpoint PATH` | auto-find latest | Use a specific checkpoint |
| `--out PATH` | `work_dirs/<dataset>/<model>/cases.png` | Output PNG |

Auto-finds the latest checkpoint for the `(dataset, model)`. Works for every
model, including spatiotemporal/graph and covariate-mode ones (pass a node index
via `--channel`).

## See also

`docs/en/experiments.md`, `docs/en/configs.md` (sweep semantics), and the
`inspect`, `run`, `sweep`, `aggregate`, `rank` skills.
