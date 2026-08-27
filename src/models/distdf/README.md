---
name: "DistDF"
implementation: rewrite
summary: "DistDF is a distribution-alignment training objective for multivariate time-series forecasting. Rather than minimising pointwise squared error, it aligns the joint distribution of forecast and label sequences via a tractable joint-distribution Wasserstein discrepancy that provably upper-bounds the harder conditional discrepancy. The method is model-agnostic and can be applied on top of diverse base forecasters to improve accuracy."
paper:
  title: "DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment"
  venue: "ICLR 2026"
  year: 2026
  url: "https://arxiv.org/abs/2510.24574"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# DistDF

DistDF is a distribution-alignment training objective for multivariate time-series forecasting. Rather than minimising pointwise squared error, it aligns the joint distribution of forecast and label sequences via a tractable joint-distribution Wasserstein discrepancy that provably upper-bounds the harder conditional discrepancy. The method is model-agnostic and can be applied on top of diverse base forecasters to improve accuracy.

<!-- model-card:canonical:start -->
## Method overview

DistDF is a distribution-alignment training objective for multivariate time-series forecasting.

## Core architecture

Rather than minimising pointwise squared error, it aligns the joint distribution of forecast and label sequences via a tractable joint-distribution Wasserstein discrepancy that provably upper-bounds the harder conditional discrepancy. The method is model-agnostic and can be applied on top of diverse base forecasters to improve accuracy.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.24574); title: DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment; venue/year: ICLR 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/DistDF.toml`](../../../configs/models/DistDF.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.24574

## Abstract
Training time-series forecasting models requires aligning the conditional distribution of model forecasts with that of the label sequence. The standard direct forecast (DF) approach resorts to minimizing the conditional negative log-likelihood, typically estimated by the mean squared error. However, this estimation proves biased when the label sequence exhibits autocorrelation. In this paper, we propose DistDF, which achieves alignment by minimizing a distributional discrepancy between the conditional distributions of forecast and label sequences. Since such conditional discrepancies are difficult to estimate from finite time-series observations, we introduce a joint-distribution Wasserstein discrepancy for time-series forecasting, which provably upper bounds the conditional discrepancy of interest. The proposed discrepancy is tractable, differentiable, and readily compatible with gradient-based optimization. Extensive experiments show that DistDF improves diverse forecasting models and achieves leading performance.

## In ModernTSF
Default config: `configs/models/DistDF.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@misc{wang2025distdf,
  author        = {Hao Wang and
                  Licheng Pan and
                  Yuan Lu and
                  Zhixuan Chu and
                  Xiaoxi Li and
                  Shuting He and
                  Zhichao Chen and
                  Haoxuan Li and
                  Qingsong Wen and
                  Zhouchen Lin},
  title         = {DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment},
  year          = {2025},
  eprint        = {2510.24574},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.24574}
}
```
