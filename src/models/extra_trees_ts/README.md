---
name: "ExtraTreesTS"
implementation: rewrite
summary: "ExtraTreesTS is a time-series forecasting adapter that wraps the Extremely Randomized Trees (Extra-Trees) ensemble method inside the ModernTSF PyTorch training harness. It applies the Extra-Trees regressor — an ensemble of decision trees with randomised split thresholds — to the sliding-window forecasting task, treating each prediction horizon step as an independent regression target."
paper:
  title: "Extremely Randomized Trees"
  venue: "Machine Learning 2006"
  year: 2006
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# ExtraTreesTS

ExtraTreesTS is a time-series forecasting adapter that wraps the Extremely Randomized Trees (Extra-Trees) ensemble method inside the ModernTSF PyTorch training harness. It applies the Extra-Trees regressor — an ensemble of decision trees with randomised split thresholds — to the sliding-window forecasting task, treating each prediction horizon step as an independent regression target.

<!-- model-card:canonical:start -->
## Method overview

ExtraTreesTS is a time-series forecasting adapter that wraps the Extremely Randomized Trees (Extra-Trees) ensemble method inside the ModernTSF PyTorch training harness.

## Core architecture

It applies the Extra-Trees regressor — an ensemble of decision trees with randomised split thresholds — to the sliding-window forecasting task, treating each prediction horizon step as an independent regression target.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Extremely Randomized Trees; venue/year: Machine Learning 2006 / 2006
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ExtraTreesTS.toml`](../../../configs/models/ExtraTreesTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=24`, `tree_depth=2`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Extremely Randomized Trees
- **Venue**: Machine Learning 2006
- **Published**: 2006
- **arXiv**: N/A

## Abstract
Extremely Randomized Trees (Extra-Trees) is a tree-based ensemble learning method introduced by Geurts, Ernst, and Wehenkel (2006). Like Random Forests, it builds an ensemble of unpruned decision or regression trees from the full training set, but with two key differences that increase randomisation: (1) split points are chosen uniformly at random within each feature's range rather than by optimising an impurity criterion, and (2) all training samples are used for building each tree (no bootstrap). These two choices trade a small increase in bias for a substantial reduction in variance and a significant speedup in training. The method consistently achieves competitive accuracy with Random Forests and gradient-boosted trees across regression and classification benchmarks, while being considerably faster to train.

## In ModernTSF
Default config: `configs/models/ExtraTreesTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{geurts2006extremely,
  author  = {Pierre Geurts and Damien Ernst and Louis Wehenkel},
  title   = {Extremely Randomized Trees},
  journal = {Machine Learning},
  volume  = {63},
  number  = {1},
  pages   = {3--42},
  year    = {2006},
  doi     = {10.1007/s10994-006-6226-1}
}
```
