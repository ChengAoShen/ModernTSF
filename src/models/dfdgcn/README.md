---
model: "DFDGCN"
forecasting_setting: "spatiotemporal"
config: "configs/models/DFDGCN.toml"
registry: "models.dfdgcn.registry"
paper_title: "Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting"
venue: "arXiv preprint"
year: 2023
arxiv: "https://arxiv.org/abs/2312.11933"
---
# DFDGCN

DFDGCN is a spatiotemporal learning model for node-structured graph data. It captures spatial dependencies in transportation networks by learning dynamic graphs in the frequency domain, mitigating time-shift effects via Fourier transform and combining identity and time embeddings with static predefined and self-adaptive graphs.

## Paper
- **Title**: Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting
- **Venue**: arXiv preprint
- **Published**: 2023 (arXiv: 2023-12)
- **arXiv**: https://arxiv.org/abs/2312.11933

## Abstract
Complex spatial dependencies in transportation networks make traffic prediction extremely challenging. Much existing work is devoted to learning dynamic graph structures among sensors, and the strategy of mining spatial dependencies from traffic data, known as data-driven, tends to be an intuitive and effective approach. However, Time-Shift of traffic patterns and noise induced by random factors hinder data-driven spatial dependence modeling. In this paper, we propose a novel dynamic frequency domain graph convolution network (DFDGCN) to capture spatial dependencies. Specifically, we mitigate the effects of time-shift by Fourier transform, and introduce the identity embedding of sensors and time embedding when capturing data for graph learning since traffic data with noise is not entirely reliable. The graph is combined with static predefined and self-adaptive graphs during graph convolution to predict future traffic data through classical causal convolutions. Extensive experiments on four real-world datasets demonstrate that our model is effective and outperforms the baselines.

## In ModernTSF
Default config: `configs/models/DFDGCN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
