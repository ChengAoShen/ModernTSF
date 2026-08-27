---
name: "RidgeRegressionTS"
implementation: rewrite
summary: "RidgeRegressionTS is a PyTorch-native adapter that implements ridge regression (L2-regularized linear regression) as a time series forecasting model, mapping a lagged feature window to the prediction horizon through a learned linear projection with L2 weight penalty, running through the standard ModernTSF trainer and supporting GPU acceleration."
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
# RidgeRegressionTS

RidgeRegressionTS is a PyTorch-native adapter that implements ridge regression (L2-regularized linear regression) as a time series forecasting model, mapping a lagged feature window to the prediction horizon through a learned linear projection with L2 weight penalty, running through the standard ModernTSF trainer and supporting GPU acceleration.

<!-- model-card:canonical:start -->
## Method overview

RidgeRegressionTS is a PyTorch-native adapter that implements ridge regression (L2-regularized linear regression) as a time series forecasting model, mapping a lagged feature window to the prediction horizon through a learned linear projection with L2 weight penalty, running through the standard ModernTSF trainer and supporting GPU acceleration.

## Core architecture

RidgeRegressionTS is a PyTorch-native adapter that implements ridge regression (L2-regularized linear regression) as a time series forecasting model, mapping a lagged feature window to the prediction horizon through a learned linear projection with L2 weight penalty, running through the standard ModernTSF trainer and supporting GPU acceleration.

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
[`configs/models/RidgeRegressionTS.toml`](../../../configs/models/RidgeRegressionTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0001`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Ridge regression is a classical regularized linear model that extends ordinary least squares by adding an L2 penalty on the regression coefficients (Tikhonov regularization). Applied to time series forecasting, the model treats lagged values of all channels as input features and predicts the future horizon via a single linear layer whose weights are regularized to avoid overfitting. The L2 penalty shrinks large coefficients toward zero, improving generalization on high-dimensional or correlated feature sets. In the ModernTSF context, the model is implemented as a `torch.nn.Module` with a learnable linear layer and a configurable regularization strength, enabling GPU-accelerated training through the standard benchmark trainer alongside all other model classes.

## In ModernTSF
Default config: `configs/models/RidgeRegressionTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{hoerl1970ridge,
  author  = {Arthur E. Hoerl and Robert W. Kennard},
  title   = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  journal = {Technometrics},
  volume  = {12},
  number  = {1},
  pages   = {55--67},
  year    = {1970},
  doi     = {10.1080/00401706.1970.10488634}
}
```
