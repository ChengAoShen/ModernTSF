---
model: "HMformer"
forecasting_setting: "time_series"
config: "configs/models/HMformer.toml"
registry: "models.hmformer.registry"
paper_title: "HMformer: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling"
venue: "AAAI 2026"
year: 2026
arxiv: ""
---
# HMformer

HMformer is a Transformer-based multivariate time-series forecasting model that proposes a hierarchical multi-scale framework to overcome the limitations of the original Transformer architecture when applied to real-world time series with complex multi-scale periodicities. It employs a hierarchical cross-scale mixing mechanism, a scale-adaptive feature expansion design, and a multi-branch complementary prediction strategy to capture intricate multi-scale temporal dynamics while retaining the Transformer's strength in modeling long-range dependencies.

## Paper
- **Title**: HMformer: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling
- **Venue**: AAAI 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
Time series forecasting plays a critical role across a wide range of domains. Recently, an increasing number of Transformer-based forecasting models have emerged, achieving remarkably competitive performance. However, real-world time series data often exhibit complex multi-scale periodicities, which are not well-suited for modeling by the original Transformer architecture originally developed for NLP tasks. To address this limitation, we propose the Hierarchical Multi-scale Time Series Transformer (HMformer), employing a novel and sophisticated framework specifically designed for multi-scale time series forecasting. Specifically, HMformer incorporates a hierarchical cross-scale mixing mechanism that progressively aggregates temporal information from fine to coarse granularities, a scale-adaptive feature expansion design enhancing the extraction of high-level temporal semantics, and a multi-branch complementary prediction strategy for effectively integrating diverse temporal patterns. Collectively, these components enable HMformer to capture intricate, multi-scale temporal dynamics while retaining the Transformer's inherent strength in modeling long-range dependencies. Extensive experiments conducted on multiple real-world benchmark datasets—encompassing both long-term and short-term forecasting tasks—demonstrate that HMformer achieves state-of-the-art performance.

## In ModernTSF
Default config: `configs/models/HMformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
