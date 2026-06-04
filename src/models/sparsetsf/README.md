---
model: "SparseTSF"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/SparseTSF.toml"
registry: "models.sparsetsf.registry"
paper_title: "SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters"
venue: "ICML 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2405.00946"
---
# SparseTSF

SparseTSF is an extremely lightweight model for long-term time series forecasting that achieves competitive performance with fewer than 1,000 parameters. Its core innovation is the Cross-Period Sparse Forecasting technique, which decouples periodicity and trend by downsampling the original sequence so that the model focuses on cross-period trend prediction rather than point-wise temporal modelling.

## Paper
- **Title**: SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters
- **Venue**: ICML 2024
- **Published**: 2024 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.00946

## Abstract
This paper introduces SparseTSF, a novel, extremely lightweight model for Long-term Time Series Forecasting (LTSF), designed to address the challenges of modeling complex temporal dependencies over extended horizons with minimal computational resources. At the heart of SparseTSF lies the Cross-Period Sparse Forecasting technique, which simplifies the forecasting task by decoupling the periodicity and trend in time series data. This technique involves downsampling the original sequences to focus on cross-period trend prediction, effectively extracting periodic features while minimizing the model's complexity and parameter count. Based on this technique, the SparseTSF model uses fewer than 1k parameters to achieve competitive or superior performance compared to state-of-the-art models. Furthermore, SparseTSF showcases remarkable generalization capabilities, making it well-suited for scenarios with limited computational resources, small samples, or low-quality data. The code is publicly available at this repository.

## In ModernTSF
Default config: `configs/models/SparseTSF.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
