# One-click experiments

ModernTSF turns ablations, hyperparameter searches, and case visualizations into
single commands. The first two are driven by the **sweep** mechanism in the run
config (a cartesian product expanded at load time — see
[configs.md](configs.md)); the third is a small plotting tool.

Preview any sweep before launching (run count, datasets, models, pred lengths):

```bash
uv run python tool/inspect_config.py --config configs/runs/<your_sweep>.toml
```

Run a sweep (single process) or batch several configs on a GPU:

```bash
uv run modern-tsf --config configs/runs/<your_sweep>.toml
# or, sequentially on a chosen GPU:
uv run python tool/tsf.py run configs/runs/<your_sweep>.toml --gpus 0
```

Every run writes a row to `work_dirs/<dataset>/<model>/performance.csv`; aggregate
and rank them with `tool/aggregate_results.py` / `tool/rank_models.py`.

---

## 1. Ablation experiments

An ablation toggles a model's **components** and compares the result. Because each
component is a field in `[model.params]`, you sweep those fields with `[sweep]`:

```toml
# configs/runs/ablation_dlinear.toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[sweep]
# Each listed value becomes a separate run; the cartesian product is expanded.
model.params.individual = [true, false]      # per-channel head on/off
model.params.kernel_size = [13, 25, 49]      # decomposition window

[sweep.task]
pred_len = [96, 336]
```

`inspect_config.py` reports this as `2 × 3 × 2 = 12` runs. You can also ablate by
**swapping whole model variants** (e.g. DLinear vs NLinear vs RLinear) with
`[sweep.extend]` — see `configs/runs/sweep_model.toml`.

## 2. Hyperparameter experiments

Identical mechanism — vary the numeric/architectural hyperparameters instead of
on/off flags:

```toml
# configs/runs/hparam_patchtst.toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/PatchTST.toml"]

[sweep]
model.params.d_model = [128, 256, 512]
model.params.n_heads = [4, 8]
training.optimizer.params.lr = [0.0001, 0.0005]

[sweep.task]
pred_len = [96, 192, 336, 720]
```

For multi-dataset / multi-model / multi-seed grids, see `configs/runs/multi_sweep.toml`
(`datasets × models × seeds × pred_len`) and `configs/runs/sweep_data.toml`
(one model across all datasets). Set `[evaluation] enable_profile = true` to also
record params/MACs per run.

## 3. Case visualization

Plot a model's forecast against the ground truth for a few test windows. Train the
model first (so a checkpoint exists), then:

```bash
# Train (writes work_dirs/<dataset>/<model>/checkpoints/<run_id>/best_checkpoint.pth)
uv run modern-tsf --config configs/runs/run_single_data.toml

# Visualize — auto-finds the latest checkpoint for that (dataset, model)
uv run python tool/visualize_predictions.py \
    --config configs/runs/run_single_data.toml --num-samples 4 --channel -1
```

This writes `work_dirs/<dataset>/<model>/cases.png` with, per sample, the input
history, the true future, and the predicted future for one channel/node. Flags:
`--num-samples N`, `--channel I` (`-1` = last channel), `--checkpoint PATH` (use a
specific checkpoint), `--out PATH`. Works for every model, including the
spatiotemporal/graph and covariate-mode ones (pass a node index via `--channel`).

---

See also: [configs.md](configs.md) (sweep semantics), [aggregate-results.md](aggregate-results.md),
[rank-models.md](rank-models.md), [plot-bubble.md](plot-bubble.md).
