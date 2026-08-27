---
name: "KalmanFilterTS"
implementation: rewrite
summary: "KalmanFilterTS is a PyTorch-native time series forecasting baseline that implements a Kalman-filter-inspired alpha-beta smoother with learnable update gains, wrapped as a standard `nn.Module` so it can be trained end-to-end through the unified ModernTSF training loop on CPU, CUDA, or MPS."
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
# KalmanFilterTS

KalmanFilterTS is a PyTorch-native time series forecasting baseline that implements a Kalman-filter-inspired alpha-beta smoother with learnable update gains, wrapped as a standard `nn.Module` so it can be trained end-to-end through the unified ModernTSF training loop on CPU, CUDA, or MPS.

<!-- model-card:canonical:start -->
## Method overview

KalmanFilterTS is a PyTorch-native time series forecasting baseline that implements a Kalman-filter-inspired alpha-beta smoother with learnable update gains, wrapped as a standard `nn.Module` so it can be trained end-to-end through the unified ModernTSF training loop on CPU, CUDA, or MPS.

## Core architecture

KalmanFilterTS is a PyTorch-native time series forecasting baseline that implements a Kalman-filter-inspired alpha-beta smoother with learnable update gains, wrapped as a standard `nn.Module` so it can be trained end-to-end through the unified ModernTSF training loop on CPU, CUDA, or MPS.

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
[`configs/models/KalmanFilterTS.toml`](../../../configs/models/KalmanFilterTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
The Kalman Filter is a classical recursive Bayesian algorithm introduced by Rudolf Kalman in 1960 that estimates the state of a linear dynamical system from noisy observations. It operates via a predict-update cycle: the predict step propagates the current state estimate forward using a transition model, and the update step incorporates a new observation, weighting predicted vs. observed values via the Kalman gain. The alpha-beta filter is a simplified fixed-gain variant that smooths position and velocity estimates. In ModernTSF, KalmanFilterTS wraps this concept in a learnable PyTorch module where the gain parameters are optimized during training, giving the classical smoothing approach the ability to adapt to each dataset while retaining its interpretable recursive structure.

## In ModernTSF
Default config: `configs/models/KalmanFilterTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{kalman1960new,
  author  = {Rudolf E. Kalman},
  title   = {A New Approach to Linear Filtering and Prediction Problems},
  journal = {Journal of Basic Engineering},
  volume  = {82},
  number  = {1},
  pages   = {35--45},
  year    = {1960},
  doi     = {10.1115/1.3662552}
}
```
