---
model: "AGCRN"
forecasting_setting: "spatiotemporal"
config: "configs/models/AGCRN.toml"
registry: "models.agcrn.registry"
paper_title: "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
venue: "NeurIPS 2020"
year: 2020
arxiv: "https://arxiv.org/abs/2007.02842"
---
# AGCRN

AGCRN (Adaptive Graph Convolutional Recurrent Network) is a spatiotemporal learning model designed for node-structured or graph-structured data. It enhances standard Graph Convolutional Networks with two adaptive modules — Node Adaptive Parameter Learning (NAPL) and Data Adaptive Graph Generation (DAGG) — and wraps them inside a recurrent architecture to jointly capture node-specific spatial patterns and temporal dynamics without requiring any pre-defined graph structure.

## Paper
- **Title**: Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting
- **Venue**: NeurIPS 2020
- **Published**: 2020 (arXiv: 2020-07)
- **arXiv**: https://arxiv.org/abs/2007.02842

## Abstract
Modeling complex spatial and temporal correlations in the correlated time series data is indispensable for understanding the traffic dynamics and predicting the future status of an evolving traffic system. Recent works focus on designing complicated graph neural network architectures to capture shared patterns with the help of pre-defined graphs. In this paper, we argue that learning node-specific patterns is essential for traffic forecasting while the pre-defined graph is avoidable. To this end, we propose two adaptive modules for enhancing Graph Convolutional Network (GCN) with new capabilities: 1) a Node Adaptive Parameter Learning (NAPL) module to capture node-specific patterns; 2) a Data Adaptive Graph Generation (DAGG) module to infer the inter-dependencies among different traffic series automatically. We further propose an Adaptive Graph Convolutional Recurrent Network (AGCRN) to capture fine-grained spatial and temporal correlations in traffic series automatically based on the two modules and recurrent networks. Our experiments on two real-world traffic datasets show AGCRN outperforms state-of-the-art by a significant margin without pre-defined graphs about spatial connections.

## In ModernTSF
Default config: `configs/models/AGCRN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
