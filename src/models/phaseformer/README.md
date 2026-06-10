---
model: "PhaseFormer"
forecasting_setting: "time_series"
config: "configs/models/PhaseFormer.toml"
registry: "models.phaseformer.registry"
paper_title: "PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting"
venue: "ICLR 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2510.04134"
---
# PhaseFormer

PhaseFormer is an efficient time series forecasting model for standard univariate and multivariate prediction. It introduces a phase perspective for exploiting periodicity: instead of treating individual patches as tokens (which incurs large parameter counts), PhaseFormer groups time steps into compact phase embeddings aligned to the dominant period and uses a lightweight routing mechanism for cross-phase interaction, achieving state-of-the-art performance with approximately 1k parameters across benchmark datasets.

## Paper
- **Title**: PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.04134

## Abstract
Periodicity is a fundamental characteristic of time series data and has long played a central role in forecasting. Recent deep learning methods strengthen the exploitation of periodicity by treating patches as basic tokens, thereby improving predictive effectiveness. However, their efficiency remains a bottleneck due to large parameter counts and heavy computational costs. This paper provides, for the first time, a clear explanation of why patch-level processing is inherently inefficient, supported by strong evidence from real-world data. To address these limitations, we introduce a phase perspective for modeling periodicity and present an efficient yet effective solution, PhaseFormer. PhaseFormer features phase-wise prediction through compact phase embeddings and efficient cross-phase interaction enabled by a lightweight routing mechanism. Extensive experiments demonstrate that PhaseFormer achieves state-of-the-art performance with around 1k parameters, consistently across benchmark datasets. Notably, it excels on large-scale and complex datasets, where models with comparable efficiency often struggle. This work marks a significant step toward truly efficient and effective time series forecasting.

## In ModernTSF
Default config: `configs/models/PhaseFormer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
