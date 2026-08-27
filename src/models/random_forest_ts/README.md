---
name: "RandomForestTS"
implementation: rewrite
summary: "RandomForestTS is a PyTorch-native adapter that applies the random forest ensemble strategy to multivariate time series forecasting. It implements a differentiable soft-tree ensemble — multiple randomized decision trees whose outputs are averaged — operating on lagged input windows, and runs through the standard ModernTSF trainer on CPU, CUDA, or MPS devices."
paper:
  title: "Random Forests"
  venue: "Machine Learning, 2001"
  year: 2001
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# RandomForestTS

RandomForestTS is a PyTorch-native adapter that applies the random forest ensemble strategy to multivariate time series forecasting. It implements a differentiable soft-tree ensemble — multiple randomized decision trees whose outputs are averaged — operating on lagged input windows, and runs through the standard ModernTSF trainer on CPU, CUDA, or MPS devices.

<!-- model-card:canonical:start -->
## Method overview

RandomForestTS is a PyTorch-native adapter that applies the random forest ensemble strategy to multivariate time series forecasting.

## Core architecture

It implements a differentiable soft-tree ensemble — multiple randomized decision trees whose outputs are averaged — operating on lagged input windows, and runs through the standard ModernTSF trainer on CPU, CUDA, or MPS devices.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Random Forests; venue/year: Machine Learning, 2001 / 2001
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/RandomForestTS.toml`](../../../configs/models/RandomForestTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Random Forests
- **Venue**: Machine Learning, 2001
- **Published**: 2001
- **arXiv**: N/A

## Abstract
Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The generalization error of a forest of tree classifiers depends on the strength of the individual trees in the forest and the correlation between them. Using a random selection of features to split each node yields error rates that compare favorably to Adaboost, but are more robust with respect to noise. Internal estimates monitor error, strength, and correlation and these are used to show the response to increasing the number of features used in the splitting. Internal estimates are also used to measure variable importance. These ideas are also applicable to regression.

## In ModernTSF
Default config: `configs/models/RandomForestTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{breiman2001random,
  author  = {Leo Breiman},
  title   = {Random Forests},
  journal = {Machine Learning},
  volume  = {45},
  number  = {1},
  pages   = {5--32},
  year    = {2001},
  doi     = {10.1023/A:1010933404324}
}
```
