# Probabilistic forecasting (`output_type`)

Point vs. probabilistic is a separate **output** axis, orthogonal to
`task.mode`. `task.mode` (`time_series` / `spatiotemporal` / `covariate`, see
[task-modes.md](task-modes.md)) selects the *data setting* — how a batch is
shaped and what the model receives. `output_type` selects what the model's
`forward` *returns* — a single point forecast, or a forecast with calibrated
uncertainty. The two compose freely: a probabilistic model can in principle
be built for any `task.mode`, though every model shipped today
(`QuantileDLinear`, `QuantilePatchTST`, `MQRNN`, `TiRex`, `GaussianMLP`,
`DeepAR`) targets `time_series`.

The pipeline reads the axis via `getattr(model, "output_type", "point")` and
only branches when it is not `"point"`. Point models never set the attribute,
so they take the unchanged default path — adding probabilistic support did
not touch the ~170 existing point models.

## The three `output_type` values

| `output_type` | `forward` returns | Meaning |
|---|---|---|
| `"point"` (default) | `(B, pred_len, C)` | A single value per step/channel. |
| `"quantile"` | `(B, pred_len, C, Q)` | A grid of `Q = len(quantile_levels)` quantiles, ascending and non-crossing along the last axis, ordered to match `quantile_levels`. |
| `"distribution"` | `(B, pred_len, C, 2)` | `(loc, scale)` of a distribution (currently `distribution_family = "gaussian"`; `scale > 0`). |

`C = c_out = 1 if features == "MS" else enc_in` — the same channel convention
as point models. A probabilistic model slices to the target channel(s) before
building the trailing `Q`/`2` axis.

### The monotone `QuantileHead`

Quantile models should not hand-roll a quantile head — they wrap the shared
`QuantileHead` in `src/models/_quantile_head.py`. Given a per-step base
feature tensor `(B, L, C, in_features)`, it projects a median anchor
(`anchor_proj`) plus strictly non-negative offsets per quantile gap
(`softplus(offset_proj(base))`), then builds the quantiles above the median by
a cumulative sum upward and below the median by a cumulative sum downward.
Because every gap is `>= 0` by construction, the output is **non-crossing by
construction**, regardless of the learned weights — output `[..., m]` (the
level closest to `0.5`) equals the anchor, and the array is monotone
non-decreasing along the last axis.

`quantile_levels` must be ascending or `QuantileHead.__init__` raises, since
the loss and metrics index the trailing axis positionally as `levels[i]`.

Templates using `QuantileHead`: `src/models/quantile_dlinear/` (minimal,
~40 lines, wraps DLinear), `src/models/quantile_patchtst/` (wraps a
transformer backbone), `src/models/mqrnn/` and `src/models/tirex/`
(from-scratch RNN / TiRex adapters).

## Loss pairing

Each `output_type` is trained with a matching loss, registered in
`src/benchmark/losses_prob.py` and selected via `[training] loss = "..."`
(point losses `mse`/`mae`/`l1` are unchanged and remain the default):

| `output_type` | `[training] loss` | Loss module | Formula |
|---|---|---|---|
| `"quantile"` | `"quantile"` | `QuantileLoss` | Weighted pinball loss: for level `q`, `max(q * (y - yhat_q), (q - 1) * (y - yhat_q))`, averaged over batch, horizon, channels, and levels. |
| `"distribution"` | `"nll_gaussian"` | `GaussianNLLLoss` | Gaussian NLL: `0.5 * log(2*pi*scale^2) + 0.5 * ((y - loc) / scale)^2`, mean-reduced; `scale` is clamped to `>= eps` (default `1e-6`). |

Both losses accept the rank-4 prediction tensor and the ordinary rank-3
target `(B, pred_len, C)` that the trainer already slices via
`_slice_pred_target`.

### `quantile_levels`: the single source of truth

`evaluation.quantile_levels` (`src/benchmark/config/schema/evaluation.py`) is
the canonical list of quantile levels — default the nine deciles
`[0.1, 0.2, ..., 0.9]`. It is threaded into two places by `run_one`:

- **Model construction**: if a model's `__init__` signature has a
  `quantile_levels` parameter (checked via `inspect.signature`) and the config
  doesn't already set `model.params.quantile_levels`, `run_one` injects
  `list(config.evaluation.quantile_levels)` automatically, so the
  `QuantileHead`'s `Q` always matches the configured levels.
- **`training.loss_params`**: when `loss == "quantile"` and
  `quantile_levels` isn't already present in `loss_params`, `run_one` injects
  it into a *copy* of `loss_params` before constructing the criterion.

Point models never see this field; `evaluation.quantile_levels` is ignored
unless the model or loss opts in.

## Metrics for probabilistic runs

`collect_prob_metrics` (`src/benchmark/evaluation/metrics.py`) computes
exactly four metrics for any non-point run, given the raw prediction tensor,
the rank-3 target, the configured `levels`, and the model's `output_type`:

| Metric | Direction | Computation |
|---|---|---|
| `crps` | lower is better | Closed-form Gaussian CRPS when `output_type == "distribution"` and `distribution_family == "gaussian"`; otherwise a quantile approximation `(2/Q) * sum_q mean(pinball_q)` over the quantile grid. |
| `wql` | lower is better | GIFT-Eval/GluonTS-style weighted quantile loss: `(1/Q) * sum_q ( 2 * sum(pinball_q) / sum|y| )`. For the quantile path this equals `crps / mean|y|`. |
| `coverage_80` | diagnostic — closer to `0.8` is better, no ranking direction enforced | Fraction of true values falling inside `[q_0.1, q_0.9]` (the central 80% band, from the `0.1`/`0.9` quantile levels — falls back to the lowest/highest configured level if `0.1`/`0.9` aren't present). |
| `width_80` | lower is better (all else equal) | Mean width of the `[q_0.1, q_0.9]` band. |

For a `"distribution"` model, `collect_prob_metrics` first builds an internal
quantile grid from `(loc, scale)` at the configured `levels` before computing
`wql`/`coverage_80`/`width_80` (CRPS alone uses the closed-form Gaussian
formula).

Probabilistic metrics run **alongside** the usual point metrics
(`mae`/`mse`/`rmse`/...), which are computed on the **median** quantile (for
`"quantile"` models) or the **`loc`** (for `"distribution"` models), so
probabilistic and point models stay comparable on the same leaderboard.
For a probabilistic model (`output_type != "point"`), `run_one` always keeps
`crps`/`wql`/`coverage_80`/`width_80` in the `performance.csv` row regardless
of `[evaluation] metrics` — listing them explicitly (as the example below
does) is only useful for making them show up first / documenting intent, not
required for them to survive.

## Example configs

Model configs only declare architecture params — the `output_type` and loss
pairing live in the model's `model.py` and the run config, respectively.

`configs/models/QuantileDLinear.toml` (quantile):

```toml
[model]
name = "QuantileDLinear"

[model.params]
enc_in = 7
kernel_size = 25
individual = false
```

`configs/models/GaussianMLP.toml` (distribution):

```toml
[model]
name = "GaussianMLP"

[model.params]
enc_in = 7
hidden_size = 256
num_layers = 2
dropout = 0.1
```

A run/smoke config wires the loss and metrics on top of any such model config
(pattern shown in the `probabilistic-forecasting` skill, following the same
`extends` style as `configs/runs/smoke_crib.toml`):

```toml
extends = ["../base.toml", "../datasets/smoke.toml", "../models/QuantileDLinear.toml"]

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

Other shipped probabilistic models follow the same two pairings:
`QuantilePatchTST`, `MQRNN`, and `TiRex` (`output_type = "quantile"`, loss
`"quantile"`), `DeepAR` (`output_type = "distribution"`, loss
`"nll_gaussian"`).

## Adding a new probabilistic model

Use the **`probabilistic-forecasting`** skill, not the plain `add-model` flow
(which builds a point model). It walks through:

- Declaring `self.output_type = "quantile"` (wrap a backbone with
  `QuantileHead`) or `self.output_type = "distribution"` +
  `self.distribution_family = "gaussian"` (emit `(loc, scale)` with
  `scale = softplus(...) + eps`).
- For `"quantile"` models only: declaring a
  `quantile_levels: list[float] | None = None` parameter in both
  `Model.__init__` and the model's `schema.py` so `run_one`'s auto-injection
  (above) applies and sizes the `QuantileHead`'s output. `"distribution"`
  models (e.g. `GaussianMLP`, `DeepAR`) don't take this parameter — they
  always emit `(loc, scale)` regardless of `quantile_levels`, which is only
  consulted by the evaluator when building `wql`/`coverage_80`/`width_80`.
- Wiring the model exactly like any other model — `registry.py`, the
  `MODEL_NAME_MAP` entry in `src/benchmark/registry/models.py`,
  `configs/models/<Name>.toml`, and a smoke config — see
  [add-model.md](add-model.md) for that base flow.
- Selecting the matching loss (`"quantile"` / `"nll_gaussian"`) and listing
  the four probabilistic metric names in `[evaluation] metrics`.
- Verifying with `uv run python tool/tsf.py smoke --model <Name>`: PASS means
  the run trains and emits finite `crps`/`wql`/`coverage_80`/`width_80` (plus
  the point metrics), with `wql < ~1`, `width_80 > 0`, `coverage_80 in [0, 1]`
  as a sanity check.

Key files:

| What | Where |
|---|---|
| Monotone quantile head | `src/models/_quantile_head.py` |
| Probabilistic losses | `src/benchmark/losses_prob.py` |
| Probabilistic metrics | `src/benchmark/evaluation/metrics.py` (`collect_prob_metrics`) |
| `output_type` gating | `src/benchmark/runner/{trainer,evaluator,run_one}.py` |
| `quantile_levels` config | `src/benchmark/config/schema/evaluation.py` |
| Templates | `src/models/{quantile_dlinear,quantile_patchtst,mqrnn,tirex,gaussian_mlp,deepar,gaussian_process_ts}/` |
