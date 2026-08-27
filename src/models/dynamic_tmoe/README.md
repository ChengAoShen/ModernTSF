---
name: "DynamicTMoE"
implementation: rewrite
summary: "DynamicTMoE is a drift-aware dynamic Mixture-of-Experts framework for non-stationary multivariate time series forecasting in the standard time-series setting. It overcomes the rigidity of traditional MoE architectures by using Maximum Mean Discrepancy (MMD) to detect distribution shifts, and dynamically expanding or pruning a heterogeneous expert pool at runtime — allowing the model to continuously adapt its capacity to changing data distributions. ModernTSF registers a lightweight native adapter that follows the shared prediction interface and normalization path from `src/adapters/recent_tsf.py`."
paper:
  title: "Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting"
  venue: "ICML 2026"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# DynamicTMoE

DynamicTMoE is a drift-aware dynamic Mixture-of-Experts framework for non-stationary multivariate time series forecasting in the standard time-series setting. It overcomes the rigidity of traditional MoE architectures by using Maximum Mean Discrepancy (MMD) to detect distribution shifts, and dynamically expanding or pruning a heterogeneous expert pool at runtime — allowing the model to continuously adapt its capacity to changing data distributions. ModernTSF registers a lightweight native adapter that follows the shared prediction interface and normalization path from `src/adapters/recent_tsf.py`.

<!-- model-card:canonical:start -->
## Method overview

DynamicTMoE is a drift-aware dynamic Mixture-of-Experts framework for non-stationary multivariate time series forecasting in the standard time-series setting.

## Core architecture

It overcomes the rigidity of traditional MoE architectures by using Maximum Mean Discrepancy (MMD) to detect distribution shifts, and dynamically expanding or pruning a heterogeneous expert pool at runtime — allowing the model to continuously adapt its capacity to changing data distributions. ModernTSF registers a lightweight native adapter that follows the shared prediction interface and normalization path from `src/adapters/recent_tsf.py`.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting; venue/year: ICML 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/DynamicTMoE.toml`](../../../configs/models/DynamicTMoE.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting
- **Venue**: ICML 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
Dynamic TMoE introduces an adaptive Mixture of Experts framework designed for time series forecasting in non-stationary environments. The method uses Maximum Mean Discrepancy (MMD) to detect distribution shifts and responds by dynamically expanding or pruning a heterogeneous expert pool, overcoming the rigidity of traditional fixed-capacity MoE designs. A drift-aware routing mechanism selects or allocates experts based on detected statistical changes in the input distribution, enabling robust forecasting under concept drift. The framework was accepted as a poster at the Forty-third International Conference on Machine Learning (ICML 2026) and demonstrates notable improvements in MSE and MAE across nine standard benchmarks compared to prior state-of-the-art methods. The official implementation is available at https://github.com/andone-07/Dynamic-TMoE.

## In ModernTSF
Default config: `configs/models/DynamicTMoE.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@misc{zhu2026dynamictmoe,
  author        = {Jiawen Zhu and Shuhan Liu and Di Weng and Yingcai Wu},
  title         = {Dynamic TMoE: {A} Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting},
  year          = {2026},
  eprint        = {2605.20678},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.20678}
}
```
