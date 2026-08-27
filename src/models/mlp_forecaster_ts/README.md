---
name: "MLPForecasterTS"
implementation: rewrite
summary: "MLPForecasterTS is a classical Multi-Layer Perceptron (MLP) baseline for time series forecasting, serving the standard univariate and multivariate prediction setting. It applies a stack of fully-connected layers with optional channel mixing and RevIN normalization to a fixed look-back window of lagged values, projecting directly to the desired forecast horizon. The model is implemented as a native PyTorch `nn.Module` adapter within the ModernTSF `_ml_tsf` family, meaning it runs on CPU, CUDA, or MPS through the standard training loop."
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
# MLPForecasterTS

MLPForecasterTS is a classical Multi-Layer Perceptron (MLP) baseline for time series forecasting, serving the standard univariate and multivariate prediction setting. It applies a stack of fully-connected layers with optional channel mixing and RevIN normalization to a fixed look-back window of lagged values, projecting directly to the desired forecast horizon. The model is implemented as a native PyTorch `nn.Module` adapter within the ModernTSF `_ml_tsf` family, meaning it runs on CPU, CUDA, or MPS through the standard training loop.

<!-- model-card:canonical:start -->
## Method overview

MLPForecasterTS is a classical Multi-Layer Perceptron (MLP) baseline for time series forecasting, serving the standard univariate and multivariate prediction setting.

## Core architecture

It applies a stack of fully-connected layers with optional channel mixing and RevIN normalization to a fixed look-back window of lagged values, projecting directly to the desired forecast horizon. The model is implemented as a native PyTorch `nn.Module` adapter within the ModernTSF `_ml_tsf` family, meaning it runs on CPU, CUDA, or MPS through the standard training loop.

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
[`configs/models/MLPForecasterTS.toml`](../../../configs/models/MLPForecasterTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
MLPForecasterTS is a foundational feedforward neural network baseline for time series forecasting. A Multi-Layer Perceptron (MLP) stacks multiple fully-connected linear layers with non-linear activations to learn a direct mapping from a fixed-length historical window of input values to a future prediction window. In the ModernTSF setting, the model operates channel-independently or with optional cross-channel mixing and applies Reversible Instance Normalization (RevIN) to stabilize training across datasets with varying scales. As a classical deep learning baseline, it serves as a simple yet non-trivial reference point for evaluating more sophisticated sequence modeling architectures.

## In ModernTSF
Default config: `configs/models/MLPForecasterTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{rumelhart1986learning,
  author  = {David E. Rumelhart and Geoffrey E. Hinton and Ronald J. Williams},
  title   = {Learning Representations by Back-Propagating Errors},
  journal = {Nature},
  volume  = {323},
  number  = {6088},
  pages   = {533--536},
  year    = {1986},
  doi     = {10.1038/323533a0}
}
```
