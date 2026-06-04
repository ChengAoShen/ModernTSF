---
model: "FiLM"
forecasting_setting: "time_series"
config: "configs/models/FiLM.toml"
registry: "models.film.registry"
paper_title: "FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting"
venue: "NeurIPS 2022"
year: 2022
arxiv: "https://arxiv.org/abs/2205.08897"
---
# FiLM

FiLM (Frequency improved Legendre Memory) is a time-series forecasting model for the standard univariate and multivariate long-term forecasting setting. It applies Legendre polynomial projections to compress and approximate historical context, applies a Fourier-domain projection to remove high-frequency noise, and uses a low-rank approximation to reduce computation — yielding a plug-in representation module that can also enhance other deep learning forecasters.

## Paper
- **Title**: FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting
- **Venue**: NeurIPS 2022
- **Published**: 2022 (arXiv: 2022-05)
- **arXiv**: https://arxiv.org/abs/2205.08897

## Abstract
Recent studies have shown that deep learning models such as RNNs and Transformers have brought significant performance gains for long-term forecasting of time series because they effectively utilize historical information. We found, however, that there is still great room for improvement in how to preserve historical information in neural networks while avoiding overfitting to noise present in the history. Addressing this allows better utilization of the capabilities of deep learning models. To this end, we design a Frequency improved Legendre Memory model, or FiLM: it applies Legendre polynomial projections to approximate historical information, uses Fourier projection to remove noise, and adds a low-rank approximation to speed up computation. Our empirical studies show that the proposed FiLM significantly improves the accuracy of state-of-the-art models in multivariate and univariate long-term forecasting by (19.2%, 22.6%), respectively. We also demonstrate that the representation module developed in this work can be used as a general plugin to improve the long-term prediction performance of other deep learning modules.

## In ModernTSF
Default config: `configs/models/FiLM.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
