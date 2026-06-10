---
model: "PHAT"
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

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2602-00654,
  author       = {Jiaming Ma and
                  Qihe Huang and
                  Haofeng Ma and
                  Guanjun Wang and
                  Sheng Huang and
                  Zhengyang Zhou and
                  Pengkun Wang and
                  Binwu Wang and
                  Yang Wang},
  title        = {{PHAT:} Modeling Period Heterogeneity for Multivariate Time Series
                  Forecasting},
  journal      = {CoRR},
  volume       = {abs/2602.00654},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2602.00654},
  doi          = {10.48550/ARXIV.2602.00654},
  eprinttype   = {arXiv},
  eprint       = {2602.00654},
  timestamp    = {Sat, 14 Mar 2026 17:13:45 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2602-00654.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
