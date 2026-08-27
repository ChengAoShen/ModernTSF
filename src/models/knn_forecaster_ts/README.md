---
name: "KNNForecasterTS"
implementation: rewrite
summary: "KNNForecasterTS is a differentiable k-nearest-neighbours style forecaster for the standard univariate and multivariate time-series setting. Instead of a hard discrete lookup, it uses a set of learnable prototype vectors and RBF (radial basis function) kernel weights to produce a soft weighted combination of prototypes, making the entire prediction end-to-end trainable with gradient descent and compatible with GPU acceleration via PyTorch."
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
# KNNForecasterTS

KNNForecasterTS is a differentiable k-nearest-neighbours style forecaster for the standard univariate and multivariate time-series setting. Instead of a hard discrete lookup, it uses a set of learnable prototype vectors and RBF (radial basis function) kernel weights to produce a soft weighted combination of prototypes, making the entire prediction end-to-end trainable with gradient descent and compatible with GPU acceleration via PyTorch.

<!-- model-card:canonical:start -->
## Method overview

KNNForecasterTS is a differentiable k-nearest-neighbours style forecaster for the standard univariate and multivariate time-series setting.

## Core architecture

Instead of a hard discrete lookup, it uses a set of learnable prototype vectors and RBF (radial basis function) kernel weights to produce a soft weighted combination of prototypes, making the entire prediction end-to-end trainable with gradient descent and compatible with GPU acceleration via PyTorch.

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
[`configs/models/KNNForecasterTS.toml`](../../../configs/models/KNNForecasterTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.08`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
K-nearest neighbours (KNN) regression is a non-parametric method that predicts an output by averaging the target values of the k training samples closest (in feature space) to the query point, using a distance metric such as Euclidean distance. Applied to time-series forecasting, KNN finds the k historical windows most similar to the current input window and uses their corresponding future segments as the forecast. The method has no single defining paper; it originates from the general KNN algorithm described by Fix & Hodges (1951) and Cover & Hart (1967). In ModernTSF, KNNForecasterTS replaces the hard discrete lookup with differentiable RBF-weighted prototypes so the model can be trained end-to-end with the standard gradient-based trainer and can run on CUDA/MPS devices.

## In ModernTSF
Default config: `configs/models/KNNForecasterTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{cover1967nearest,
  author  = {Thomas M. Cover and Peter E. Hart},
  title   = {Nearest Neighbor Pattern Classification},
  journal = {IEEE Transactions on Information Theory},
  volume  = {13},
  number  = {1},
  pages   = {21--27},
  year    = {1967},
  doi     = {10.1109/TIT.1967.1053964}
}
```
