---
name: "HMformer"
implementation: rewrite
summary: "HMformer is a Transformer-based multivariate time-series forecasting model that proposes a hierarchical multi-scale framework to overcome the limitations of the original Transformer architecture when applied to real-world time series with complex multi-scale periodicities. It employs a hierarchical cross-scale mixing mechanism, a scale-adaptive feature expansion design, and a multi-branch complementary prediction strategy to capture intricate multi-scale temporal dynamics while retaining the Transformer's strength in modeling long-range dependencies."
paper:
  title: "HMformer: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling"
  venue: "AAAI 2026"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# HMformer

HMformer is a Transformer-based multivariate time-series forecasting model that proposes a hierarchical multi-scale framework to overcome the limitations of the original Transformer architecture when applied to real-world time series with complex multi-scale periodicities. It employs a hierarchical cross-scale mixing mechanism, a scale-adaptive feature expansion design, and a multi-branch complementary prediction strategy to capture intricate multi-scale temporal dynamics while retaining the Transformer's strength in modeling long-range dependencies.

<!-- model-card:canonical:start -->
## Method overview

HMformer is a Transformer-based multivariate time-series forecasting model that proposes a hierarchical multi-scale framework to overcome the limitations of the original Transformer architecture when applied to real-world time series with complex multi-scale periodicities.

## Core architecture

It employs a hierarchical cross-scale mixing mechanism, a scale-adaptive feature expansion design, and a multi-branch complementary prediction strategy to capture intricate multi-scale temporal dynamics while retaining the Transformer's strength in modeling long-range dependencies.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: HMformer: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling; venue/year: AAAI 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/HMformer.toml`](../../../configs/models/HMformer.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: HMformer: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling
- **Venue**: AAAI 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
Time series forecasting plays a critical role across a wide range of domains. Recently, an increasing number of Transformer-based forecasting models have emerged, achieving remarkably competitive performance. However, real-world time series data often exhibit complex multi-scale periodicities, which are not well-suited for modeling by the original Transformer architecture originally developed for NLP tasks. To address this limitation, we propose the Hierarchical Multi-scale Time Series Transformer (HMformer), employing a novel and sophisticated framework specifically designed for multi-scale time series forecasting. Specifically, HMformer incorporates a hierarchical cross-scale mixing mechanism that progressively aggregates temporal information from fine to coarse granularities, a scale-adaptive feature expansion design enhancing the extraction of high-level temporal semantics, and a multi-branch complementary prediction strategy for effectively integrating diverse temporal patterns. Collectively, these components enable HMformer to capture intricate, multi-scale temporal dynamics while retaining the Transformer's inherent strength in modeling long-range dependencies. Extensive experiments conducted on multiple real-world benchmark datasets—encompassing both long-term and short-term forecasting tasks—demonstrate that HMformer achieves state-of-the-art performance.

## In ModernTSF
Default config: `configs/models/HMformer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@inproceedings{huang2026hmformer,
  author    = {Renjun Huang and Han Xiao and Bingqing Li and Baili Zhang and Jianhua Lyu},
  title     = {{HMformer}: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  url       = {https://github.com/dantian123121/HMformer}
}
```
