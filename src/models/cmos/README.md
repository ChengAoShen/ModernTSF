---
model: "CMoS"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/CMoS.toml"
registry: "models.cmos.registry"
paper_title: "CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations"
venue: "arXiv preprint"
year: 2025
arxiv: "https://arxiv.org/abs/2505.19090"
---
# CMoS

CMoS is a super-lightweight multivariate time series forecasting model for the standard time-series setting. Rather than learning shape embeddings, it directly models spatial correlations between different time-series chunks using a Correlation Mixing strategy that captures diverse channel dependencies with minimal parameters, and an optional Periodicity Injection technique for faster convergence — achieving competitive accuracy at up to 100x the parameter efficiency of DLinear.

## Paper
- **Title**: CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.19090

## Abstract
Recent advances in lightweight time series forecasting models suggest the inherent simplicity of time series forecasting tasks. In this paper, we present CMoS, a super-lightweight time series forecasting model. Instead of learning the embedding of the shapes, CMoS directly models the spatial correlations between different time series chunks. Additionally, we introduce a Correlation Mixing technique that enables the model to capture diverse spatial correlations with minimal parameters, and an optional Periodicity Injection technique to ensure faster convergence. Despite utilizing as low as 1% of the lightweight model DLinear's parameters count, experimental results demonstrate that CMoS outperforms existing state-of-the-art models across multiple datasets. Furthermore, the learned weights of CMoS exhibit great interpretability, providing practitioners with valuable insights into temporal structures within specific application scenarios.

## In ModernTSF
Default config: `configs/models/CMoS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
