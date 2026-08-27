---
name: "ARIMATS"
implementation: rewrite
summary: "ARIMATS is a PyTorch-native adapter for the classical ARIMA (Autoregressive Integrated Moving Average) family of statistical models, serving the standard time-series forecasting setting. It wraps differentiable ARIMA-inspired predictors — which estimate future values from differenced historical observations — inside the unified `torch.nn.Module` interface, enabling evaluation on the same trainer and benchmarking pipeline as deep learning models."
paper:
  title: "Time Series Analysis: Forecasting and Control"
  venue: "Holden-Day (book) / N/A (classical baseline)"
  year: 1970
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# ARIMATS

ARIMATS is a PyTorch-native adapter for the classical ARIMA (Autoregressive Integrated Moving Average) family of statistical models, serving the standard time-series forecasting setting. It wraps differentiable ARIMA-inspired predictors — which estimate future values from differenced historical observations — inside the unified `torch.nn.Module` interface, enabling evaluation on the same trainer and benchmarking pipeline as deep learning models.

<!-- model-card:canonical:start -->
## Method overview

ARIMATS is a PyTorch-native adapter for the classical ARIMA (Autoregressive Integrated Moving Average) family of statistical models, serving the standard time-series forecasting setting.

## Core architecture

It wraps differentiable ARIMA-inspired predictors — which estimate future values from differenced historical observations — inside the unified `torch.nn.Module` interface, enabling evaluation on the same trainer and benchmarking pipeline as deep learning models.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Time Series Analysis: Forecasting and Control; venue/year: Holden-Day (book) / N/A (classical baseline) / 1970
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ARIMATS.toml`](../../../configs/models/ARIMATS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Time Series Analysis: Forecasting and Control
- **Venue**: Holden-Day (book); N/A (classical baseline)
- **Published**: 1970
- **arXiv**: N/A

## Abstract
ARIMA (Autoregressive Integrated Moving Average) is a classical statistical framework for modeling and forecasting univariate time series, introduced by Box and Jenkins (1970). An ARIMA(p,d,q) model combines autoregressive terms (AR), differencing to achieve stationarity (I), and moving-average terms (MA). The model captures linear temporal dependencies by regressing the current value on its own past values and on past forecast errors, after applying d rounds of differencing to remove trend non-stationarity. Model orders (p, d, q) are typically selected via the ACF/PACF plots and information criteria such as AIC/BIC. ARIMA remains a widely used baseline for short- and medium-term forecasting across economics, meteorology, and engineering.

## In ModernTSF
Default config: `configs/models/ARIMATS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

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
