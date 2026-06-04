---
model: "PHAT"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/PHAT.toml"
registry: "models.phat.registry"
paper_title: "PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting"
venue: "arXiv preprint"
year: 2026
arxiv: "https://arxiv.org/abs/2602.00654"
---
# PHAT

PHAT (Period Heterogeneity-Aware Transformer) is a Transformer-based model for multivariate time series forecasting that explicitly models periodic heterogeneity — the fact that different variables exhibit distinct and dynamically changing periods. It organises inputs into a three-dimensional periodic bucket tensor and applies a positive-negative attention mechanism to capture both periodic alignment and periodic deviation. The ModernTSF adapter is an unverified paper reconstruction and not an official reproduction.

## Paper
- **Title**: PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2026 (arXiv: 2026-02)
- **arXiv**: https://arxiv.org/abs/2602.00654

## Abstract
While existing multivariate time series forecasting models have advanced significantly in modeling periodicity, they largely neglect the periodic heterogeneity common in real-world data, where variables exhibit distinct and dynamically changing periods. To effectively capture this periodic heterogeneity, we propose PHAT (Period Heterogeneity-Aware Transformer). Specifically, PHAT arranges multivariate inputs into a three-dimensional "periodic bucket" tensor, where the dimensions correspond to variable group characteristics with similar periodicity, time steps aligned by phase, and offsets within the period. By restricting interactions within buckets and masking cross-bucket connections, PHAT effectively avoids interference from inconsistent periods. We also propose a positive-negative attention mechanism, which captures periodic dependencies from two perspectives: periodic alignment and periodic deviation. Additionally, the periodic alignment attention scores are decomposed into positive and negative components, with a modulation term encoding periodic priors. This modulation constrains the attention mechanism to more faithfully reflect the underlying periodic trends. A mathematical explanation is provided to support this property. We evaluate PHAT comprehensively on 14 real-world datasets against 18 baselines, and the results show that it significantly outperforms existing methods, achieving highly competitive forecasting performance.

## In ModernTSF
Default config: `configs/models/PHAT.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
