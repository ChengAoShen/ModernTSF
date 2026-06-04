---
model: "STID"
category: "spatiotemporal_learning"
category_name: "Spatiotemporal Learning"
forecasting_setting: "spatiotemporal"
config: "configs/models/STID.toml"
registry: "models.stid.registry"
paper_title: "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
venue: "CIKM 2022"
year: 2022
arxiv: "https://arxiv.org/abs/2208.05233"
---
# STID

STID (Spatial-Temporal IDentity) is an MLP-based spatiotemporal forecasting model designed for node-structured or graph-structured data. It attaches learnable spatial identity embeddings (one per node) and temporal identity embeddings (time-of-day and day-of-week) to the input, then encodes all features with simple multi-layer perceptrons to predict future node values, achieving strong performance with minimal complexity.

## Paper
- **Title**: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
- **Venue**: CIKM 2022
- **Published**: 2022 (arXiv: 2022-08)
- **arXiv**: https://arxiv.org/abs/2208.05233

## Abstract
Multivariate Time Series (MTS) forecasting plays a vital role in a wide range of applications. Recently, Spatial-Temporal Graph Neural Networks (STGNNs) have become increasingly popular MTS forecasting methods due to their state-of-the-art performance. However, recent works are becoming more sophisticated with limited performance improvements. This phenomenon motivates us to explore the critical factors of MTS forecasting and design a model that is as powerful as STGNNs, but more concise and efficient. In this paper, we identify the indistinguishability of samples in both spatial and temporal dimensions as a key bottleneck, and propose a simple yet effective baseline for MTS forecasting by attaching Spatial and Temporal IDentity information (STID), which achieves the best performance and efficiency simultaneously based on simple Multi-Layer Perceptrons (MLPs). These results suggest that we can design efficient and effective models as long as they solve the indistinguishability of samples, without being limited to STGNNs.

## In ModernTSF
Default config: `configs/models/STID.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
