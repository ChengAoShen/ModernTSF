---
name: "PolynomialRegressionTS"
implementation: rewrite
summary: "PolynomialRegressionTS is a time series forecasting model for univariate and multivariate sequence prediction. It extends linear regression by constructing polynomial lag features — raw, squared, and square-root transformations of the input window — and learning a linear map from these features to the forecast horizon."
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
# PolynomialRegressionTS

PolynomialRegressionTS is a time series forecasting model for univariate and multivariate sequence prediction. It extends linear regression by constructing polynomial lag features — raw, squared, and square-root transformations of the input window — and learning a linear map from these features to the forecast horizon.

<!-- model-card:canonical:start -->
## Method overview

PolynomialRegressionTS is a time series forecasting model for univariate and multivariate sequence prediction.

## Core architecture

It extends linear regression by constructing polynomial lag features — raw, squared, and square-root transformations of the input window — and learning a linear map from these features to the forecast horizon.

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
[`configs/models/PolynomialRegressionTS.toml`](../../../configs/models/PolynomialRegressionTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Polynomial regression is a classical statistical technique that enriches the feature space of a linear model by including nonlinear transformations of the input variables. In the time-series forecasting context, the historical window values are expanded with squared and square-root lag features before a linear predictor maps them to the output horizon. This polynomial feature augmentation allows the model to capture simple nonlinear trends without the overhead of a deep neural network. The ModernTSF implementation trains this model end-to-end as a `torch.nn.Module`, enabling execution on GPU/MPS via the standard training loop and making it a useful nonlinear classical baseline alongside purely linear methods.

## In ModernTSF
Default config: `configs/models/PolynomialRegressionTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@book{draper1998applied,
  author    = {Norman R. Draper and Harry Smith},
  title     = {Applied Regression Analysis},
  edition   = {3rd},
  publisher = {Wiley},
  address   = {New York},
  year      = {1998},
  doi       = {10.1002/9781118625590}
}
```
