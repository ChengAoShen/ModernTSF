---
name: "PMDformer"
implementation: rewrite
summary: "PMDformer is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It decouples patch-level local shape fluctuations from their mean (trend) level through Patch-Mean Decoupling (PMD), combines Proximal Variable Attention (PVA) to focus on the most relevant inter-variable interactions, and applies Trend Recovery Attention (TRA) to restore long-term trend information, improving both forecasting accuracy and computational efficiency."
paper:
  title: "PMDformer: Patch-Mean Decoupling Transformer for Long-term Forecasting"
  venue: "ICLR 2026"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# PMDformer

PMDformer is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It decouples patch-level local shape fluctuations from their mean (trend) level through Patch-Mean Decoupling (PMD), combines Proximal Variable Attention (PVA) to focus on the most relevant inter-variable interactions, and applies Trend Recovery Attention (TRA) to restore long-term trend information, improving both forecasting accuracy and computational efficiency.

<!-- model-card:canonical:start -->
## Method overview

PMDformer is a Transformer-based long-term time-series forecasting model for the standard time-series setting.

## Core architecture

It decouples patch-level local shape fluctuations from their mean (trend) level through Patch-Mean Decoupling (PMD), combines Proximal Variable Attention (PVA) to focus on the most relevant inter-variable interactions, and applies Trend Recovery Attention (TRA) to restore long-term trend information, improving both forecasting accuracy and computational efficiency.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: PMDformer: Patch-Mean Decoupling Transformer for Long-term Forecasting; venue/year: ICLR 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/PMDformer.toml`](../../../configs/models/PMDformer.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: PMDformer: Patch-Mean Decoupling Transformer for Long-term Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
The official paper abstract is not available on arXiv. According to publicly available information about the ICLR 2026 accepted paper (https://github.com/aohu1105/PMDformer), PMDformer introduces three core innovations: (1) Patch-Mean Decoupling (PMD), which separates local shape fluctuations from their absolute magnitude (mean level) to reduce bias and better capture underlying patterns; (2) Proximal Variable Attention (PVA), which strengthens focus on the most relevant and temporally proximal inter-variable interactions; and (3) Trend Recovery Attention (TRA), which restores long-term trend information to improve both responsiveness and stability in forecasting. Together, these components deliver stronger forecasting accuracy and stability while reducing memory usage compared to previous patch-based Transformer methods.

## In ModernTSF
Default config: `configs/models/PMDformer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@inproceedings{hu2026pmdformer,
  author    = {Ao Hu and Liangjian Wen and Jiang Duan and Yong Dai and Yan He and Dongkai Wang and Jun Wang and Yukun Zhang and Ruoxi Jiang and Zenglin Xu},
  title     = {{PMD}former: Patch-Mean Decoupling Transformer for Long-term Forecasting},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=rfJ41gK9Ct}
}
```
