---
model: "MSGNet"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/MSGNet.toml"
registry: "models.msgnet.registry"
paper_title: "MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting"
venue: "AAAI 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2401.00423"
---
# MSGNet

MSGNet is a time series forecasting model for multivariate sequence prediction. It captures varying inter-series correlations across multiple time scales by combining frequency domain analysis (FFT-based period extraction) with an adaptive mixhop graph convolution layer, while self-attention handles intra-series dependencies within each scale — all without requiring an external adjacency matrix.

## Paper
- **Title**: MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting
- **Venue**: AAAI 2024
- **Published**: 2024 (arXiv: 2023-12)
- **arXiv**: https://arxiv.org/abs/2401.00423

## Abstract
Multivariate time series forecasting poses an ongoing challenge across various disciplines. Time series data often exhibit diverse intra-series and inter-series correlations, contributing to intricate and interwoven dependencies that have been the focus of numerous studies. Nevertheless, a significant research gap remains in comprehending the varying inter-series correlations across different time scales among multiple time series, an area that has received limited attention in the literature. To bridge this gap, this paper introduces MSGNet, an advanced deep learning model designed to capture the varying inter-series correlations across multiple time scales using frequency domain analysis and adaptive graph convolution. By leveraging frequency domain analysis, MSGNet effectively extracts salient periodic patterns and decomposes the time series into distinct time scales. The model incorporates a self-attention mechanism to capture intra-series dependencies, while introducing an adaptive mixhop graph convolution layer to autonomously learn diverse inter-series correlations within each time scale. Extensive experiments are conducted on several real-world datasets to showcase the effectiveness of MSGNet. Furthermore, MSGNet possesses the ability to automatically learn explainable multi-scale inter-series correlations, exhibiting strong generalization capabilities even when applied to out-of-distribution samples.

## In ModernTSF
Default config: `configs/models/MSGNet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
