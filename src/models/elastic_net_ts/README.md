---
name: "ElasticNetTS"
implementation: rewrite
summary: "ElasticNetTS is a time series forecasting model that applies the Elastic Net regression method — a linear predictor combining L1 (Lasso) and L2 (Ridge) regularization — to autoregressive lag-feature forecasting. It fits one linear model per channel and output step, making it an interpretable and computationally efficient baseline. The ModernTSF adapter wraps the Elastic Net as a `torch.nn.Module` so it runs within the standard training loop and can be dispatched to CUDA/MPS devices."
paper:
  title: "Regularization and Variable Selection via the Elastic Net"
  venue: "Journal of the Royal Statistical Society, Series B"
  year: 2005
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# ElasticNetTS

ElasticNetTS is a time series forecasting model that applies the Elastic Net regression method — a linear predictor combining L1 (Lasso) and L2 (Ridge) regularization — to autoregressive lag-feature forecasting. It fits one linear model per channel and output step, making it an interpretable and computationally efficient baseline. The ModernTSF adapter wraps the Elastic Net as a `torch.nn.Module` so it runs within the standard training loop and can be dispatched to CUDA/MPS devices.

<!-- model-card:canonical:start -->
## Method overview

ElasticNetTS is a time series forecasting model that applies the Elastic Net regression method — a linear predictor combining L1 (Lasso) and L2 (Ridge) regularization — to autoregressive lag-feature forecasting.

## Core architecture

It fits one linear model per channel and output step, making it an interpretable and computationally efficient baseline. The ModernTSF adapter wraps the Elastic Net as a `torch.nn.Module` so it runs within the standard training loop and can be dispatched to CUDA/MPS devices.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Regularization and Variable Selection via the Elastic Net; venue/year: Journal of the Royal Statistical Society, Series B / 2005
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ElasticNetTS.toml`](../../../configs/models/ElasticNetTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=1e-05`, `l2_penalty=0.0001`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Regularization and Variable Selection via the Elastic Net
- **Venue**: Journal of the Royal Statistical Society, Series B
- **Published**: 2005
- **arXiv**: N/A

## Abstract
Elastic Net is a regularized regression method that linearly combines the L1 and L2 penalty terms of the Lasso and Ridge methods. It was introduced by Zou and Hastie (2005) to address the limitations of Lasso — in particular its instability when features are correlated and its inability to select more variables than observations. The Elastic Net penalty encourages a grouping effect in which strongly correlated predictors tend to be selected or dropped together. This combination achieves the sparsity of Lasso and the stability of Ridge, making it well suited to high-dimensional regression and variable selection problems where predictors exhibit correlation structure.

## In ModernTSF
Default config: `configs/models/ElasticNetTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{zou2005elasticnet,
  author  = {Hui Zou and Trevor Hastie},
  title   = {Regularization and Variable Selection via the Elastic Net},
  journal = {Journal of the Royal Statistical Society: Series {B} (Statistical Methodology)},
  volume  = {67},
  number  = {2},
  pages   = {301--320},
  year    = {2005},
  doi     = {10.1111/j.1467-9868.2005.00503.x}
}
```
