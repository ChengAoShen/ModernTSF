---
name: "RandomForestTS"
implementation: rewrite
summary: "RandomForestTS is an independent differentiable forest baseline that averages soft trees with fixed random feature subspaces."
paper:
  title: "Random Forests"
  venue: "Machine Learning, 2001"
  year: 2001
  url: "https://doi.org/10.1023/A:1010933404324"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# RandomForestTS

RandomForestTS is an independent differentiable forest baseline that averages soft trees with fixed random feature subspaces.

<!-- model-card:canonical:start -->
## Method overview

RandomForestTS is an independent differentiable forest baseline that averages soft trees with fixed random feature subspaces.

## Core architecture

RandomForestTS is an independent differentiable forest baseline that averages soft trees with fixed random feature subspaces.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1023/A:1010933404324); title: Random Forests; venue/year: Machine Learning, 2001 / 2001
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/RandomForestTS.toml`](../../../configs/models/RandomForestTS.toml).

## Differences

This clean-room baseline averages independently parameterized soft trees with deterministic random feature masks. It does not bootstrap training rows, greedily fit hard splits, estimate out-of-bag error, or reproduce scikit-learn. The cited paper supplies the ensemble principle only; no external source code was inspected or copied. Evidence is in `verification/rewrite/RandomForestTS.json`.

## Shared components

- [`revin`](../../components/revin.py)
- [`soft_tree`](../../components/soft_tree.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_estimators=16`, `tree_depth=3`, `feature_fraction=0.7`, `temperature=1.0`, `random_seed=1729`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Random Forests
- **Venue**: Machine Learning, 2001
- **Published**: 2001
- **arXiv**: N/A

## Abstract
Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The generalization error of a forest of tree classifiers depends on the strength of the individual trees in the forest and the correlation between them. Using a random selection of features to split each node yields error rates that compare favorably to Adaboost, but are more robust with respect to noise. Internal estimates monitor error, strength, and correlation and these are used to show the response to increasing the number of features used in the splitting. Internal estimates are also used to measure variable importance. These ideas are also applicable to regression.

## In ModernTSF
Default config: `configs/models/RandomForestTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Verification

This clean-room baseline averages independently parameterized soft trees with deterministic random feature masks. It does not bootstrap training rows, greedily fit hard splits, estimate out-of-bag error, or reproduce scikit-learn. The cited paper supplies the ensemble principle only; no external source code was inspected or copied. Evidence is in `verification/rewrite/RandomForestTS.json`.

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
