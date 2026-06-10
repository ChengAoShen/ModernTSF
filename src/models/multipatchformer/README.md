---
model: "MultiPatchFormer"
forecasting_setting: "time_series"
config: "configs/models/MultiPatchFormer.toml"
registry: "models.multipatchformer.registry"
paper_title: "A multiscale model for multivariate time series forecasting"
venue: "Scientific Reports 2025"
year: 2025
arxiv: ""
---
# MultiPatchFormer

MultiPatchFormer is a Transformer-based time series forecasting model that integrates multi-scale patch-wise temporal modeling with channel-wise representation learning. The input time series is divided into patches at multiple resolutions to capture temporal correlations across different time granularities; a subsequent channel-wise encoder models inter-series relationships; and a multi-step linear decoder generates the final multi-horizon predictions, reducing overfitting and noise effects. It targets both univariate and multivariate long-term forecasting settings.

## Paper
- **Title**: A multiscale model for multivariate time series forecasting
- **Venue**: Scientific Reports 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Transformer based models for time-series forecasting have shown promising performance and during the past few years different Transformer variants have been proposed in time-series forecasting domain. However, most of the existing methods, mainly represent the time-series from a single scale, making it challenging to capture various time granularities or ignore inter-series correlations between the series which might lead to inaccurate forecasts. In this paper, we address the above mentioned shortcomings and propose a Transformer based model which integrates multi-scale patch-wise temporal modeling and channel-wise representation. In the multi-scale temporal part, the input time-series is divided into patches of different resolutions to capture temporal correlations associated with various scales. The channel-wise encoder which comes after the temporal encoder, models the relations among the input series to capture the intricate interactions between them. In our framework, we further design a multi-step linear decoder to generate the final predictions for the purpose of reducing over-fitting and noise effects. Extensive experiments on seven real world datasets indicate that our model (MultiPatchFormer) achieves state-of-the-art results by surpassing other current baseline models in terms of error metrics and shows stronger generalizability.

## In ModernTSF
Default config: `configs/models/MultiPatchFormer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{naghashi2025multiscale,
  author  = {Vahid Naghashi and Mounir Boukadoum and Abdoulaye Banire Diallo},
  title   = {A Multiscale Model for Multivariate Time Series Forecasting},
  journal = {Scientific Reports},
  volume  = {15},
  number  = {1},
  year    = {2025},
  doi     = {10.1038/s41598-024-82417-4},
  url     = {https://doi.org/10.1038/s41598-024-82417-4}
}
```
