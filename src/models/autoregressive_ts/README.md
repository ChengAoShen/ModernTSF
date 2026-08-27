---
name: "AutoRegressiveTS"
implementation: rewrite
summary: "AutoRegressiveTS is a classical autoregressive lag model for univariate and multivariate time-series forecasting. It directly maps the historical input window to the future prediction window using a learned linear projection over lagged observations, and is wrapped as a PyTorch `nn.Module` so that it integrates with the standard ModernTSF training loop and can run on CUDA/MPS devices."
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
# AutoRegressiveTS

AutoRegressiveTS is a classical autoregressive lag model for univariate and multivariate time-series forecasting. It directly maps the historical input window to the future prediction window using a learned linear projection over lagged observations, and is wrapped as a PyTorch `nn.Module` so that it integrates with the standard ModernTSF training loop and can run on CUDA/MPS devices.

<!-- model-card:canonical:start -->
## Method overview

AutoRegressiveTS is a classical autoregressive lag model for univariate and multivariate time-series forecasting.

## Core architecture

It directly maps the historical input window to the future prediction window using a learned linear projection over lagged observations, and is wrapped as a PyTorch `nn.Module` so that it integrates with the standard ModernTSF training loop and can run on CUDA/MPS devices.

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
[`configs/models/AutoRegressiveTS.toml`](../../../configs/models/AutoRegressiveTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Autoregressive (AR) models predict the next value (or block of values) in a time series as a linear combination of a fixed number of past observations, known as the lag order. The parameters are typically estimated by ordinary least squares or Yule–Walker equations. When extended to the vector setting (VAR), each variable is regressed on its own lags and the lags of all other variables. The AR/VAR family is one of the oldest and most studied approaches in time-series analysis, forming the basis for more complex models such as ARIMA and state-space methods. In ModernTSF the model is implemented as a differentiable linear layer that maps the full input window to the full prediction horizon in a single forward pass, enabling end-to-end gradient-based training.

## In ModernTSF
Default config: `configs/models/AutoRegressiveTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@book{box1970time,
  author    = {George E. P. Box and Gwilym M. Jenkins},
  title     = {Time Series Analysis: Forecasting and Control},
  publisher = {Holden-Day},
  address   = {San Francisco},
  year      = {1970},
  url       = {https://archive.org/details/timeseriesanalys0000boxg}
}
```
