---
name: "GradientBoostingTS"
implementation: rewrite
summary: "GradientBoostingTS is a PyTorch-native adapter that applies gradient boosting regression to multivariate time series forecasting. It uses a residual ensemble of soft decision trees with linear base learners, trained end-to-end through the standard ModernTSF trainer, and can operate on CPU, CUDA, or MPS devices."
paper:
  title: "Greedy function approximation: A gradient boosting machine"
  venue: "Annals of Statistics, 2001"
  year: 2001
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# GradientBoostingTS

GradientBoostingTS is a PyTorch-native adapter that applies gradient boosting regression to multivariate time series forecasting. It uses a residual ensemble of soft decision trees with linear base learners, trained end-to-end through the standard ModernTSF trainer, and can operate on CPU, CUDA, or MPS devices.

<!-- model-card:canonical:start -->
## Method overview

GradientBoostingTS is a PyTorch-native adapter that applies gradient boosting regression to multivariate time series forecasting.

## Core architecture

It uses a residual ensemble of soft decision trees with linear base learners, trained end-to-end through the standard ModernTSF trainer, and can operate on CPU, CUDA, or MPS devices.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Greedy function approximation: A gradient boosting machine; venue/year: Annals of Statistics, 2001 / 2001
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/GradientBoostingTS.toml`](../../../configs/models/GradientBoostingTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=12`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Greedy function approximation: A gradient boosting machine
- **Venue**: Annals of Statistics, 2001
- **Published**: 2001
- **arXiv**: N/A

## Abstract
Function estimation/approximation is viewed from the perspective of numerical optimization in function space, rather than parameter space. A connection between stagewise additive expansions and steepest-descent minimization is identified. A general gradient descent "boosting" paradigm is developed for additive expansions based on any fitting criterion. Special enhancements are derived for regression with squared error loss, absolute error loss, and huberized M-loss, with applications to least-squares, least absolute deviation, and Huber-M loss functions for regression, and multiclass logistic likelihood for classification. Regression trees are shown to be especially amenable to this approach, giving rise to the Gradient Tree Boosting procedure. Competitive statistical performance of the resulting procedures is demonstrated on several datasets, producing highly robust, interpretable nonparametric regression and classification models appropriate for data mining applications.

## In ModernTSF
Default config: `configs/models/GradientBoostingTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{friedman2001greedy,
  author  = {Jerome H. Friedman},
  title   = {Greedy Function Approximation: A Gradient Boosting Machine},
  journal = {The Annals of Statistics},
  volume  = {29},
  number  = {5},
  pages   = {1189--1232},
  year    = {2001},
  doi     = {10.1214/aos/1013203451}
}
```
