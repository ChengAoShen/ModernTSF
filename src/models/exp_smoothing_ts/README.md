---
name: "ExpSmoothingTS"
implementation: rewrite
summary: "ExpSmoothingTS is a PyTorch-native time series forecasting adapter that implements an exponential-smoothing-inspired predictor for the standard time series forecasting setting. It uses learned decay weights to progressively downweight older observations, extrapolates trends from the smoothed history, and runs through the ModernTSF standard trainer so it can be evaluated on GPU/CPU alongside deep learning models."
paper:
  title: ""
  venue: "N/A (classical baseline)"
  year: null
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# ExpSmoothingTS

ExpSmoothingTS is a PyTorch-native time series forecasting adapter that implements an exponential-smoothing-inspired predictor for the standard time series forecasting setting. It uses learned decay weights to progressively downweight older observations, extrapolates trends from the smoothed history, and runs through the ModernTSF standard trainer so it can be evaluated on GPU/CPU alongside deep learning models.

<!-- model-card:canonical:start -->
## Method overview

ExpSmoothingTS is a PyTorch-native time series forecasting adapter that implements an exponential-smoothing-inspired predictor for the standard time series forecasting setting.

## Core architecture

It uses learned decay weights to progressively downweight older observations, extrapolates trends from the smoothed history, and runs through the ModernTSF standard trainer so it can be evaluated on GPU/CPU alongside deep learning models.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: not available; venue/year: N/A (classical baseline) / not available
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ExpSmoothingTS.toml`](../../../configs/models/ExpSmoothingTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Exponential smoothing is a classical family of time series forecasting methods that assign exponentially decreasing weights to past observations, placing the most emphasis on recent data. Simple exponential smoothing forecasts a constant level, while double (Holt) and triple (Holt-Winters) variants additionally model additive or multiplicative trend and seasonality components via additional smoothing parameters. The ExpSmoothingTS adapter in ModernTSF re-implements the core smoothing idea as a differentiable PyTorch module with learnable decay parameters, enabling the classical technique to be trained end-to-end with gradient descent and deployed on the same hardware as neural forecasting models.

## In ModernTSF
Default config: `configs/models/ExpSmoothingTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{holt2004forecasting,
  author  = {Charles C. Holt},
  title   = {Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages},
  journal = {International Journal of Forecasting},
  volume  = {20},
  number  = {1},
  pages   = {5--10},
  year    = {2004},
  doi     = {10.1016/j.ijforecast.2003.09.015}
}
```
