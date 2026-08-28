---
name: "PULSE"
implementation: rewrite
summary: "PULSE is a physics-informed generative framework for non-stationary forecasting. The local clean-room implementation separates phase anchors from stochastic residuals, evolves future anchors with a two-stage Phase Router, and denormalizes only the residual coordinate."
paper:
  title: "Generative Phase Evolution for Non-Stationary Time Series Forecasting"
  venue: "ICML 2026"
  year: 2026
  url: "https://arxiv.org/abs/2605.16793"
codebase:
  url: "https://github.com/Gemost/PULSE"
  revision: ""
  license: ""
  usage: reference-only
---
# PULSE

PULSE is a physics-informed generative framework for non-stationary time-series forecasting. The local implementation independently realizes its phase-anchor decomposition, dual-stream evolution, and coordinate-consistent reconstruction.

<!-- model-card:canonical:start -->
## Method overview

PULSE is a physics-informed generative framework for non-stationary forecasting.

## Core architecture

The local clean-room implementation separates phase anchors from stochastic residuals, evolves future anchors with a two-stage Phase Router, and denormalizes only the residual coordinate.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2605.16793); title: Generative Phase Evolution for Non-Stationary Time Series Forecasting; venue/year: ICML 2026 / 2026
- [codebase](https://github.com/Gemost/PULSE); revision: `not available`; license: `not available`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/PULSE.toml`](../../../configs/models/PULSE.toml).

## Differences

Clean-room implementation: confirmed.

This is a compact randomly initialized realization, not the released trained system. It implements Statistic-Aware Mixup and Frequency-MAE as explicit training utilities, but the generic forward path does not invoke them automatically. It omits timestamp covariate encoding, uses direct circular phase positions, and resamples Phase Router tokens to a fixed resolution. The reference-only repository was not inspected or copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=32`, `dropout=0.1`, `phase_period=24`, `phase_resolution=8`, `router_heads=4`, `eps=1e-05`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Generative Phase Evolution for Non-Stationary Time Series Forecasting
- **Venue**: ICML 2026
- **Published**: 2026
- **arXiv**: https://arxiv.org/abs/2605.16793

## Abstract
PULSE introduces a physics-informed framework that reframes time-series forecasting as a generative phase-evolution problem rather than a historical-fitting task. The method decomposes each series into a deterministic phase structure and stochastic residual fluctuations. Future phase trajectories are generated autoregressively, while a separate module simulates distribution shifts in the residual component, enabling the model to handle non-stationary dynamics that cause distribution shifts between training and inference. Evaluated across 12 real-world datasets covering 24 evaluation metrics, PULSE achieved the best result on 18 of 24 metrics, demonstrating strong generalization to unseen non-stationary conditions.

## Source and verification

Clean-room implementation: confirmed.

This is a compact randomly initialized realization, not the released trained system. It implements Statistic-Aware Mixup and Frequency-MAE as explicit training utilities, but the generic forward path does not invoke them automatically. It omits timestamp covariate encoding, uses direct circular phase positions, and resamples Phase Router tokens to a fixed resolution. The reference-only repository was not inspected or copied.

## In ModernTSF
Default config: `configs/models/PULSE.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Citation

```bibtex
@inproceedings{liu2026pulse,
  author    = {Yangyou Liu and Zezhi Shao and Xinyu Chen and Hu Chen and Fei Wang and Yuankai Wu},
  title     = {{PULSE}: Generative Phase Evolution for Non-Stationary Time Series Forecasting},
  booktitle = {Forty-Third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://github.com/Gemost/PULSE}
}
```
