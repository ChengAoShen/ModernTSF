---
name: probabilistic-forecasting
description: Add or run a *probabilistic* forecasting model in ModernTSF — one that outputs quantiles or a distribution (prediction intervals / uncertainty) instead of a single point, and is scored with CRPS / WQL / coverage. Use this whenever the task involves prediction intervals, quantiles, uncertainty, a probabilistic or distributional forecast, pinball / quantile loss, Gaussian NLL, or the metrics `crps` / `wql` / `coverage_80` / `width_80` — even if the user only says "predict a range" or "give me uncertainty bands". Do NOT use the plain `add-model` flow for these: it builds a point model. Probabilistic output is a separate `output_type` axis, orthogonal to `task.mode`.
---

## The core idea: `output_type` is a separate axis

Point vs probabilistic is **not** a `task.mode` (those — `time_series` /
`spatiotemporal` / `covariate` — select the *data setting*). It is an
independent **output** axis carried by a model attribute, so a probabilistic
model composes with any `task.mode`:

- A model declares `self.output_type` in `__init__`, one of
  `"point"` (default), `"quantile"`, or `"distribution"`.
- The pipeline reads it via `getattr(model, "output_type", "point")` and only
  branches when it is not `"point"`. **The ~170 point models are untouched** —
  this mirrors the existing non-invasive `aux_loss` convention.

**Output tensor convention** (probabilistic models return rank-4):

| `output_type` | `forward` returns | `K` | trained with |
|---|---|---|---|
| `"point"` (default) | `(B, pred_len, C)` | — | `mse` / `mae` |
| `"quantile"` | `(B, pred_len, C, Q)` | `Q = len(quantile_levels)`, ascending | `quantile` (pinball) |
| `"distribution"` (gaussian) | `(B, pred_len, C, 2)` = `(loc, scale>0)` | `2` | `nll_gaussian` |

`C = c_out = 1 if features == "MS" else enc_in`. Slice to the target channel
*before* building the `K` axis (mirror the templates).

## Add a QUANTILE model (the common case)

Wrap any point backbone with the shared monotone head — do **not** hand-roll a
quantile head. `src/models/_quantile_head.py :: QuantileHead` produces a
**non-crossing** `(B, L, C, Q)` grid by construction (median anchor ± cumulative
`softplus` offsets), so quantiles can never cross regardless of weights.

Fastest path: copy an existing wrapper. `src/models/quantile_dlinear/` is the
minimal template (≈40 lines); `src/models/quantile_patchtst/` wraps a
transformer; `src/models/mqrnn/` is a from-scratch RNN seq2seq.

```python
# src/models/<name>/model.py
from models.<backbone>.model import Model as Backbone
from models._quantile_head import QuantileHead

_DEFAULT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, features="M",
                 quantile_levels=None, **backbone_kw):
        super().__init__()
        self.pred_len, self.features = pred_len, features
        self.c_out = 1 if features == "MS" else enc_in
        self.output_type = "quantile"                       # <- declares the axis
        levels = list(quantile_levels) if quantile_levels else _DEFAULT_LEVELS
        self.backbone = Backbone(seq_len=seq_len, pred_len=pred_len, ...)
        self.quantile_head = QuantileHead(levels, in_features=1)

    def forward(self, x, *args):
        base = self.backbone(x)                # (B, pred_len, enc_in)
        if self.features == "MS":
            base = base[:, :, -1:]
        base = base[:, -self.pred_len:, :]      # (B, pred_len, c_out)
        return self.quantile_head(base.unsqueeze(-1))   # (B, pred_len, c_out, Q)
```

Declare `quantile_levels: list[float] | None = None` in both `Model.__init__`
**and** the model's `schema.py` — `run_one` auto-injects
`evaluation.quantile_levels` into any model whose `__init__` accepts it (via
`inspect.signature`), so the head's `Q` always matches the configured levels.

## Add a DISTRIBUTION model

Emit per-step parameters; set `output_type = "distribution"` and
`distribution_family = "gaussian"`. Templates: `src/models/gaussian_mlp/`
(fresh) and `src/models/deepar/` (RNN). The scale must be strictly positive —
apply `softplus(...) + eps`.

```python
self.output_type = "distribution"
self.distribution_family = "gaussian"
# forward -> torch.stack([loc, F.softplus(raw_scale) + 1e-6], dim=-1)  # (B, pred_len, c_out, 2)
```

## Wire it up (same as any model)

`registry.py` `register()` with the `(cfg, params)` factory, a `MODEL_NAME_MAP`
entry in `src/benchmark/registry/models.py`, `configs/models/<Name>.toml`, and a
`configs/runs/smoke_<module>.toml`. See the `add-model` skill for the scaffold;
the only probabilistic-specific parts are the `output_type` attribute and the
loss/metrics below. Add a `README.md` card (front matter) so `understand-model`
can surface it.

## Losses & metrics (already built — just select them)

- **Losses** live in `src/benchmark/losses_prob.py`: `quantile` (weighted
  pinball over the levels) and `nll_gaussian` (Gaussian NLL on `(loc, scale)`).
  Select via `[training] loss = "quantile"` (or `"nll_gaussian"`). Point losses
  are unchanged and remain the default.
- **Metrics** live in `src/benchmark/evaluation/metrics.py ::
  collect_prob_metrics`: `crps`, `wql` (GIFT-Eval weighted quantile loss),
  `coverage_80` and `width_80` (the central [q0.1, q0.9] interval). They are
  computed automatically for any non-point model; the 10 point metrics are still
  reported on the median (quantile) / `loc` (distribution) so prob models stay
  comparable. **List the prob metric names in `[evaluation] metrics`** so they
  survive the run's metric whitelist.

## Run / smoke recipe

A probabilistic run config (`configs/runs/smoke_<module>.toml`, CPU, 1 epoch):

```toml
extends = ["../base.toml", "../datasets/smoke.toml", "../models/<Name>.toml"]
[task]
seq_len = 96
pred_len = 12
features = "M"
[training]
loss = "quantile"          # or "nll_gaussian" for a distribution model
[evaluation]
metrics = ["crps", "wql", "coverage_80", "width_80", "mae", "mse"]
# quantile_levels defaults to the 9 deciles; override here if needed
```

```bash
uv run python tool/tsf.py smoke --model <Name>   # globs smoke_<module>*.toml
```

PASS means it trains and emits finite `crps/wql/coverage_80/width_80` (+ point
metrics). Sanity: `wql` should be < ~1, `width_80 > 0`, `coverage_80 ∈ [0, 1]`.

## Key rules & gotchas

- **Never break the point path.** Gate everything on `output_type`; the default
  `"point"` arm must stay byte-identical. Verify with an existing point smoke
  (e.g. `smoke_dlinear`).
- **`quantile_levels` must be ascending** — `QuantileHead` raises otherwise (the
  loss / metrics index that axis positionally as `levels[i]`).
- **Known limitation**: the wrappers feed only the scalar point into
  `QuantileHead` (`in_features=1`), so intervals are wide / under-calibrated and
  `coverage_80` saturates on under-trained models. Routing the backbone's
  per-step hidden vector (`in_features=d_model`) is the calibration upgrade.
- **CRPS**: closed-form for gaussian distributions, quantile approximation
  otherwise — both already handled in `collect_prob_metrics`.

## Key files

| What | Where |
|---|---|
| Monotone quantile head | `src/models/_quantile_head.py` |
| Probabilistic losses | `src/benchmark/losses_prob.py` |
| Probabilistic metrics | `src/benchmark/evaluation/metrics.py` (`collect_prob_metrics`) |
| `output_type` gating | `src/benchmark/runner/{trainer,evaluator,run_one}.py` |
| `quantile_levels` config | `src/benchmark/config/schema/evaluation.py` |
| Templates | `src/models/{quantile_dlinear,quantile_patchtst,mqrnn,gaussian_mlp,deepar}/` |
