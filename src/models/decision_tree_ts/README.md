---
name: "DecisionTreeTS"
implementation: rewrite
summary: "DecisionTreeTS is a PyTorch-native adapter that wraps a decision-tree-style predictor over flattened lag features for univariate and multivariate time series forecasting. It registers under the standard ModernTSF trainer interface, allowing the tree-based computation to run on CPU, CUDA, or MPS tensors."
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
# DecisionTreeTS

DecisionTreeTS is a PyTorch-native adapter that wraps a decision-tree-style predictor over flattened lag features for univariate and multivariate time series forecasting. It registers under the standard ModernTSF trainer interface, allowing the tree-based computation to run on CPU, CUDA, or MPS tensors.

<!-- model-card:canonical:start -->
## Method overview

DecisionTreeTS is a PyTorch-native adapter that wraps a decision-tree-style predictor over flattened lag features for univariate and multivariate time series forecasting.

## Core architecture

It registers under the standard ModernTSF trainer interface, allowing the tree-based computation to run on CPU, CUDA, or MPS tensors.

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
[`configs/models/DecisionTreeTS.toml`](../../../configs/models/DecisionTreeTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=1`, `tree_depth=4`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Decision trees are classical non-parametric supervised learning models that recursively partition the input feature space using axis-aligned splits, selecting the split at each node by minimising an impurity criterion (e.g., mean squared error for regression). For time series forecasting, the model is applied by constructing a feature matrix of lagged input values and training a separate tree (or a single multi-output tree) to predict each future step. Although decision trees are highly interpretable and require no gradient-based optimisation, they can overfit without regularisation (maximum depth, minimum samples per leaf) and do not naturally capture sequential structure. In ModernTSF they are wrapped as a differentiable-style torch.nn.Module for uniform pipeline integration.

## In ModernTSF
Default config: `configs/models/DecisionTreeTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@book{DBLP:books/wa/BreimanFOS84,
  author       = {Leo Breiman and
                  J. H. Friedman and
                  Richard A. Olshen and
                  C. J. Stone},
  title        = {Classification and Regression Trees},
  publisher    = {Wadsworth},
  year         = {1984},
  isbn         = {0-534-98053-8},
  timestamp    = {Mon, 10 Jul 2023 12:50:10 +0200},
  biburl       = {https://dblp.org/rec/books/wa/BreimanFOS84.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
