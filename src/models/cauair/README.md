---
model: "CauAir"
forecasting_setting: "covariate"
config: "configs/models/CauAir.toml"
registry: "models.cauair.registry"
paper_title: "Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting"
venue: "IJCAI 2025"
year: 2025
arxiv: ""
---
# CauAir

CauAir is a covariate prediction model originally designed for nationwide air quality forecasting. It explicitly models the causal association between weather covariates and air quality indices (AQI) through a Transformer-based architecture called CachLormer, which replaces standard attention with a cache-attention mechanism that captures covariate-AQI causality in a coarse-grained manner, enabling competitive performance at low computational cost across many nodes.

## Paper
- **Title**: Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting
- **Venue**: IJCAI 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Air quality prediction plays a crucial role in the development of smart cities, garnering significant attention from both academia and industry. Current air quality prediction models encounter two major limitations: their high computational complexity limits scalability to nationwide datasets, and they often regard weather covariates as optional auxiliary information. In reality, weather covariates can have a substantial impact on air quality indices (AQI), exhibiting a significant causal association. In this paper, we first present a nationwide air quality dataset to address the lack of open-source, large-scale datasets in this field. Then we propose a causal learning model, CauAir, for air quality prediction that harnesses the powerful representation capabilities of the Transformer to explicitly model the causal association between weather covariates and AQI. To address the high complexity of traditional Transformers, we design CachLormer, which features two key innovations: a simplified architecture with redundant components removed, and a cache-attention mechanism that employs learnable embeddings for perceiving causal association between AQI and weather covariates in a coarse-grained perspective. We use information theory to illustrate the superiority of the proposed model. Finally, experimental results on three datasets with 28 as the baseline demonstrate that our model achieves competitive performance, while maintaining high training efficiency and low memory consumption.

## In ModernTSF
Default config: `configs/models/CauAir.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
