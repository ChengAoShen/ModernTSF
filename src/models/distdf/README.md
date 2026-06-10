---
model: "DistDF"
forecasting_setting: "time_series"
config: "configs/models/DistDF.toml"
registry: "models.distdf.registry"
paper_title: "DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment"
venue: "ICLR 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2510.24574"
---
# DistDF

DistDF is a distribution-alignment training objective for multivariate time-series forecasting. Rather than minimising pointwise squared error, it aligns the joint distribution of forecast and label sequences via a tractable joint-distribution Wasserstein discrepancy that provably upper-bounds the harder conditional discrepancy. The method is model-agnostic and can be applied on top of diverse base forecasters to improve accuracy.

## Paper
- **Title**: DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.24574

## Abstract
Training time-series forecasting models requires aligning the conditional distribution of model forecasts with that of the label sequence. The standard direct forecast (DF) approach resorts to minimizing the conditional negative log-likelihood, typically estimated by the mean squared error. However, this estimation proves biased when the label sequence exhibits autocorrelation. In this paper, we propose DistDF, which achieves alignment by minimizing a distributional discrepancy between the conditional distributions of forecast and label sequences. Since such conditional discrepancies are difficult to estimate from finite time-series observations, we introduce a joint-distribution Wasserstein discrepancy for time-series forecasting, which provably upper bounds the conditional discrepancy of interest. The proposed discrepancy is tractable, differentiable, and readily compatible with gradient-based optimization. Extensive experiments show that DistDF improves diverse forecasting models and achieves leading performance.

## In ModernTSF
Default config: `configs/models/DistDF.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
