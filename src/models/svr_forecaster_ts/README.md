---
name: "SVRForecasterTS"
implementation: rewrite
summary: "SVRForecasterTS is a PyTorch-native time series forecasting adapter inspired by Support Vector Regression (SVR). It uses RBF (radial basis function) prototype support vectors and a linear residual head to produce multi-step forecasts, wrapped in the standard ModernTSF `torch.nn.Module` interface so it can be trained with gradient descent and run on CPU, CUDA, or MPS hardware alongside deep learning models."
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
# SVRForecasterTS

SVRForecasterTS is a PyTorch-native time series forecasting adapter inspired by Support Vector Regression (SVR). It uses RBF (radial basis function) prototype support vectors and a linear residual head to produce multi-step forecasts, wrapped in the standard ModernTSF `torch.nn.Module` interface so it can be trained with gradient descent and run on CPU, CUDA, or MPS hardware alongside deep learning models.

<!-- model-card:canonical:start -->
## Method overview

SVRForecasterTS is a PyTorch-native time series forecasting adapter inspired by Support Vector Regression (SVR).

## Core architecture

It uses RBF (radial basis function) prototype support vectors and a linear residual head to produce multi-step forecasts, wrapped in the standard ModernTSF `torch.nn.Module` interface so it can be trained with gradient descent and run on CPU, CUDA, or MPS hardware alongside deep learning models.

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
[`configs/models/SVRForecasterTS.toml`](../../../configs/models/SVRForecasterTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=48`, `kernel_gamma=0.05`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Support Vector Regression (SVR) is a classical kernel-based supervised learning method derived from Support Vector Machines. Given a set of training examples, SVR seeks a function that deviates from the true target values by at most a margin epsilon while remaining as flat as possible. Predictions are expressed as a weighted sum of kernel evaluations (commonly the RBF kernel) between the query point and a sparse subset of training examples called support vectors. SVRForecasterTS re-implements this kernel regression idea as a differentiable PyTorch module: learnable RBF prototype centers replace the classical SVM solver, and a linear residual layer corrects systematic bias. This allows the classical SVR concept to be trained end-to-end with gradient descent and evaluated on GPU or MPS hardware within the ModernTSF benchmark pipeline.

## In ModernTSF
Default config: `configs/models/SVRForecasterTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@inproceedings{drucker1996support,
  author    = {Harris Drucker and Christopher J. C. Burges and Linda Kaufman and Alexander J. Smola and Vladimir Vapnik},
  title     = {Support Vector Regression Machines},
  booktitle = {Advances in Neural Information Processing Systems 9 (NIPS 1996)},
  pages     = {155--161},
  year      = {1996},
  url       = {https://proceedings.neurips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html}
}
```
