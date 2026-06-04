---
model: "StemGNN"
category: "spatiotemporal_learning"
category_name: "Spatiotemporal Learning"
forecasting_setting: "spatiotemporal"
config: "configs/models/StemGNN.toml"
registry: "models.stemgnn.registry"
paper_title: "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting"
venue: "NeurIPS 2020"
year: 2020
arxiv: "https://arxiv.org/abs/2103.07719"
---
# StemGNN

StemGNN (Spectral Temporal Graph Neural Network) is a spatiotemporal model for multivariate time-series forecasting that captures inter-series correlations and temporal dependencies jointly in the spectral domain. It combines a Graph Fourier Transform (GFT) for spatial correlation and a Discrete Fourier Transform (DFT) for temporal patterns in a unified end-to-end framework, learning the inter-series graph structure automatically from data without pre-defined priors.

## Paper
- **Title**: Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting
- **Venue**: NeurIPS 2020
- **Published**: 2020 (arXiv: 2021-03)
- **arXiv**: https://arxiv.org/abs/2103.07719

## Abstract
Multivariate time-series forecasting plays a crucial role in many real-world applications. It is a challenging problem as one needs to consider both intra-series temporal correlations and inter-series correlations simultaneously. Recently, there have been multiple works trying to capture both correlations, but most, if not all of them only capture temporal correlations in the time domain and resort to pre-defined priors as inter-series relationships. In this paper, we propose Spectral Temporal Graph Neural Network (StemGNN) to further improve the accuracy of multivariate time-series forecasting. StemGNN captures inter-series correlations and temporal dependencies jointly in the spectral domain. It combines Graph Fourier Transform (GFT) which models inter-series correlations and Discrete Fourier Transform (DFT) which models temporal dependencies in an end-to-end framework. After passing through GFT and DFT, the spectral representations hold clear patterns and can be predicted effectively by convolution and sequential learning modules. Moreover, StemGNN learns inter-series correlations automatically from the data without using pre-defined priors. We conduct extensive experiments on ten real-world datasets to demonstrate the effectiveness of StemGNN. Code is available at https://github.com/microsoft/StemGNN/

## In ModernTSF
Default config: `configs/models/StemGNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
