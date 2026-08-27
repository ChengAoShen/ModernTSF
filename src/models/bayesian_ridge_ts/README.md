---
name: "BayesianRidgeTS"
implementation: rewrite
summary: "BayesianRidgeTS is a time series forecasting model for univariate and multivariate sequence prediction. It is a PyTorch-native linear predictor inspired by Bayesian ridge regression, applying stronger shrinkage regularisation over the input window to produce forecasts for the prediction horizon."
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
# BayesianRidgeTS

BayesianRidgeTS is a time series forecasting model for univariate and multivariate sequence prediction. It is a PyTorch-native linear predictor inspired by Bayesian ridge regression, applying stronger shrinkage regularisation over the input window to produce forecasts for the prediction horizon.

<!-- model-card:canonical:start -->
## Method overview

BayesianRidgeTS is a time series forecasting model for univariate and multivariate sequence prediction.

## Core architecture

It is a PyTorch-native linear predictor inspired by Bayesian ridge regression, applying stronger shrinkage regularisation over the input window to produce forecasts for the prediction horizon.

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
[`configs/models/BayesianRidgeTS.toml`](../../../configs/models/BayesianRidgeTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.05`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0005`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Bayesian ridge regression is a classical statistical technique that places a Gaussian prior over the regression weights, equivalent to L2 (ridge) regularisation with a prior variance determined by empirical Bayes or cross-validation. In the time-series setting each output channel is predicted independently by a linear map from the flattened input window; the Bayesian prior encourages compact, well-regularised weight matrices that generalise better under limited data. The ModernTSF implementation trains this model end-to-end as a `torch.nn.Module`, enabling use on GPU/MPS via the standard training loop and making it a strong classical baseline for comparison against deep forecasters.

## In ModernTSF
Default config: `configs/models/BayesianRidgeTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{mackay1992bayesian,
  author  = {David J. C. MacKay},
  title   = {Bayesian Interpolation},
  journal = {Neural Computation},
  volume  = {4},
  number  = {3},
  pages   = {415--447},
  year    = {1992},
  doi     = {10.1162/neco.1992.4.3.415},
  url     = {https://doi.org/10.1162/neco.1992.4.3.415}
}
```
