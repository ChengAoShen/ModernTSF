---
name: "PULSE"
implementation: rewrite
summary: "PULSE is a physics-informed generative framework for non-stationary time-series forecasting. Instead of passively fitting historical patterns, it separates deterministic phase structures from stochastic fluctuations, generates future phase trajectories, and simulates residual distribution shifts — an approach that shifts forecasting from historical fitting to generative phase evolution. In ModernTSF, a lightweight adapter (RecentTSFModel, style=\"phase\") captures this inductive bias within the standard training pipeline."
paper:
  title: "Generative Phase Evolution for Non-Stationary Time Series Forecasting"
  venue: "ICML 2026"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# PULSE

PULSE is a physics-informed generative framework for non-stationary time-series forecasting. Instead of passively fitting historical patterns, it separates deterministic phase structures from stochastic fluctuations, generates future phase trajectories, and simulates residual distribution shifts — an approach that shifts forecasting from historical fitting to generative phase evolution. In ModernTSF, a lightweight adapter (RecentTSFModel, style="phase") captures this inductive bias within the standard training pipeline.

<!-- model-card:canonical:start -->
## Method overview

PULSE is a physics-informed generative framework for non-stationary time-series forecasting.

## Core architecture

Instead of passively fitting historical patterns, it separates deterministic phase structures from stochastic fluctuations, generates future phase trajectories, and simulates residual distribution shifts — an approach that shifts forecasting from historical fitting to generative phase evolution. In ModernTSF, a lightweight adapter (RecentTSFModel, style="phase") captures this inductive bias within the standard training pipeline.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Generative Phase Evolution for Non-Stationary Time Series Forecasting; venue/year: ICML 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/PULSE.toml`](../../../configs/models/PULSE.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Generative Phase Evolution for Non-Stationary Time Series Forecasting
- **Venue**: ICML 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
PULSE introduces a physics-informed framework that reframes time-series forecasting as a generative phase-evolution problem rather than a historical-fitting task. The method decomposes each series into a deterministic phase structure and stochastic residual fluctuations. Future phase trajectories are generated autoregressively, while a separate module simulates distribution shifts in the residual component, enabling the model to handle non-stationary dynamics that cause distribution shifts between training and inference. Evaluated across 12 real-world datasets covering 24 evaluation metrics, PULSE achieved the best result on 18 of 24 metrics, demonstrating strong generalization to unseen non-stationary conditions.

## In ModernTSF
Default config: `configs/models/PULSE.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

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
