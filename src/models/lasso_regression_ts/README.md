---
name: "LassoRegressionTS"
implementation: rewrite
summary: "LassoRegressionTS is a PyTorch-native adapter that applies Lasso (L1-regularised linear) regression for time-series forecasting. It treats the look-back window as a flat lag feature vector and fits a linear projection to the prediction horizon, with L1 regularisation promoting sparsity over lag features. Running the linear layer as a `torch.nn.Module` allows training on CPU, CUDA, or MPS with the standard ModernTSF trainer."
paper:
  title: "Regression Shrinkage and Selection via the Lasso"
  venue: "Journal of the Royal Statistical Society: Series B, 1996"
  year: 1996
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# LassoRegressionTS

LassoRegressionTS is a PyTorch-native adapter that applies Lasso (L1-regularised linear) regression for time-series forecasting. It treats the look-back window as a flat lag feature vector and fits a linear projection to the prediction horizon, with L1 regularisation promoting sparsity over lag features. Running the linear layer as a `torch.nn.Module` allows training on CPU, CUDA, or MPS with the standard ModernTSF trainer.

<!-- model-card:canonical:start -->
## Method overview

LassoRegressionTS is a PyTorch-native adapter that applies Lasso (L1-regularised linear) regression for time-series forecasting.

## Core architecture

It treats the look-back window as a flat lag feature vector and fits a linear projection to the prediction horizon, with L1 regularisation promoting sparsity over lag features. Running the linear layer as a `torch.nn.Module` allows training on CPU, CUDA, or MPS with the standard ModernTSF trainer.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Regression Shrinkage and Selection via the Lasso; venue/year: Journal of the Royal Statistical Society: Series B, 1996 / 1996
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/LassoRegressionTS.toml`](../../../configs/models/LassoRegressionTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=1e-05`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Regression Shrinkage and Selection via the Lasso
- **Venue**: Journal of the Royal Statistical Society: Series B, 1996
- **Published**: 1996
- **arXiv**: N/A

## Abstract
Lasso (Least Absolute Shrinkage and Selection Operator) is a classical penalised regression method introduced by Tibshirani (1996). It minimises the residual sum of squares subject to the sum of the absolute values of the regression coefficients being less than a constant. This L1 constraint has the effect of shrinking some coefficients exactly to zero, producing sparse and interpretable models while avoiding the instability of ordinary subset selection. The method combines the variable-selection capability of subset regression with the continuous shrinkage of ridge regression, making it effective when only a small subset of predictors is truly informative. In the time-series forecasting setting, Lasso regression is applied channel-by-channel over lag features derived from the historical input window, using L1 regularisation to identify the most predictive lags for each output channel.

## In ModernTSF
Default config: `configs/models/LassoRegressionTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{tibshirani1996regression,
  author  = {Robert Tibshirani},
  title   = {Regression Shrinkage and Selection via the Lasso},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume  = {58},
  number  = {1},
  pages   = {267--288},
  year    = {1996},
  doi     = {10.1111/j.2517-6161.1996.tb02080.x}
}
```
