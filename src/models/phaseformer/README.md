---
name: "PhaseFormer"
implementation: rewrite
summary: "PhaseFormer is an efficient time series forecasting model for standard univariate and multivariate prediction. It introduces a phase perspective for exploiting periodicity: instead of treating individual patches as tokens (which incurs large parameter counts), PhaseFormer groups time steps into compact phase embeddings aligned to the dominant period and uses a lightweight routing mechanism for cross-phase interaction, achieving state-of-the-art performance with approximately 1k parameters across benchmark datasets."
paper:
  title: "PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting"
  venue: "ICLR 2026"
  year: 2026
  url: "https://arxiv.org/abs/2510.04134"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# PhaseFormer

PhaseFormer is an efficient time series forecasting model for standard univariate and multivariate prediction. It introduces a phase perspective for exploiting periodicity: instead of treating individual patches as tokens (which incurs large parameter counts), PhaseFormer groups time steps into compact phase embeddings aligned to the dominant period and uses a lightweight routing mechanism for cross-phase interaction, achieving state-of-the-art performance with approximately 1k parameters across benchmark datasets.

<!-- model-card:canonical:start -->
## Method overview

PhaseFormer is an efficient time series forecasting model for standard univariate and multivariate prediction.

## Core architecture

It introduces a phase perspective for exploiting periodicity: instead of treating individual patches as tokens (which incurs large parameter counts), PhaseFormer groups time steps into compact phase embeddings aligned to the dominant period and uses a lightweight routing mechanism for cross-phase interaction, achieving state-of-the-art performance with approximately 1k parameters across benchmark datasets.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.04134); title: PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting; venue/year: ICLR 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/PhaseFormer.toml`](../../../configs/models/PhaseFormer.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.04134

## Abstract
Periodicity is a fundamental characteristic of time series data and has long played a central role in forecasting. Recent deep learning methods strengthen the exploitation of periodicity by treating patches as basic tokens, thereby improving predictive effectiveness. However, their efficiency remains a bottleneck due to large parameter counts and heavy computational costs. This paper provides, for the first time, a clear explanation of why patch-level processing is inherently inefficient, supported by strong evidence from real-world data. To address these limitations, we introduce a phase perspective for modeling periodicity and present an efficient yet effective solution, PhaseFormer. PhaseFormer features phase-wise prediction through compact phase embeddings and efficient cross-phase interaction enabled by a lightweight routing mechanism. Extensive experiments demonstrate that PhaseFormer achieves state-of-the-art performance with around 1k parameters, consistently across benchmark datasets. Notably, it excels on large-scale and complex datasets, where models with comparable efficiency often struggle. This work marks a significant step toward truly efficient and effective time series forecasting.

## In ModernTSF
Default config: `configs/models/PhaseFormer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@misc{niu2025phaseformer,
  author        = {Yiming Niu and
                  Jinliang Deng and
                  Yongxin Tong},
  title         = {PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting},
  year          = {2025},
  eprint        = {2510.04134},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.04134}
}
```
