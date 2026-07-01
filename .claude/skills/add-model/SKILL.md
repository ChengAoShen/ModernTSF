---
name: add-model
description: Add a new model to ModernTSF by scaffolding it with `tool/tsf.py new-model`, filling the architecture, and verifying with `tsf smoke`. Use when the user wants to integrate a new PyTorch model, register a custom architecture, or wire a new forecaster into the benchmark pipeline.
---

## The fast path (3 steps)

### 1. Scaffold

Ask for the **model name** (PascalCase) and its **hyper-parameters**, then run:

```bash
# Plain (B, T, C) forecaster
uv run python tool/tsf.py new-model --name MyModel --params "enc_in:int,hidden:int=128,dropout:float=0.1"

# Node-structured graph / spatiotemporal model (reads params["adj_mx"])
uv run python tool/tsf.py new-model --name MyGraphNet --graph --params "enc_in:int,hidden:int=64"
```

`--params` is a comma list of `field:type[=default]` (types: `int|float|str|bool`).
`enc_in:int` is always added. This generates, in one shot:

- `src/models/<module>/model.py` — adapter with a shape-correct **placeholder** `forward`
- `src/models/<module>/schema.py` — Pydantic `ModelParameterConfig`
- `src/models/<module>/registry.py` — `register()` with the `(cfg, params)` factory
  (graph variant reads `params.get("adj_mx")`)
- `configs/models/<Name>.toml` — model config
- `configs/runs/smoke_<module>.toml` — end-to-end smoke run config
- the `MODEL_NAME_MAP` entry in `src/benchmark/registry/models.py`

### 2. Implement the architecture

Replace the placeholder in `src/models/<module>/model.py` `forward` with the real
model. Contract: `forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, *args)` →
plain models return `(B, pred_len, enc_in)`; graph models return `(B, pred_len, N)`.
Tune `configs/models/<Name>.toml` params as needed.

### 3. Verify end-to-end

```bash
uv run python tool/tsf.py smoke --model MyModel
```

This runs the smoke config (1 epoch, CPU) and reports `PASS`/`FAIL` with the output
shape check. Iterate on step 2 until it passes.

## Key rules

- Factory signature is `lambda cfg, params: Model(...)` — read `cfg.task.seq_len` /
  `cfg.task.pred_len` and `params["..."]`. Graph models get the adjacency from
  `params.get("adj_mx")` (the runner injects it from the dataset).
- `forward` must accept `(x, x_mark, dec_inp, dec_mark)`; use `*args` to ignore
  unused temporal marks.
- Registration is idempotent; re-running `register()` is safe.

## Manual reference

The annotated per-file templates (for hand-wiring or understanding the scaffold
output) live in `docs/en/add-model.md`.

## Advanced training objectives

Use the simplest training contract that fits the model:

- Plain forecasting: return `(B, pred_len, C)` and let the configured loss train it.
- Additive regularization: set `self.aux_loss` to a finite scalar in `forward`.
- Full custom objective: set `self.train_loss_override` in `forward`; this replaces
  the configured training criterion, while validation/early stopping still use the
  configured observation loss.
- Objectives that need the future target: declare `requires_train_target = True`
  and implement `set_train_target(self, y_or_none)`. The trainer feeds the raw
  future target only for training and clears it before validation/evaluation.
- Two-stage models: implement `pretrain(self, train_loader, device)`. It runs once
  before optimizer construction; keep smoke configs tiny for this stage.

Do not use legacy names such as `wants_target` / `set_target`. Models using any
custom-objective hook (`requires_train_target` / `set_train_target` /
`train_loss_override`) must not rely on `torch.nn.DataParallel` — the trainer
fails fast for that combination.
