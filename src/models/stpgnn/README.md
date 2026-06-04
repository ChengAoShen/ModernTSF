---
model: "STPGNN"
forecasting_setting: "spatiotemporal"
config: "configs/models/STPGNN.toml"
registry: "models.stpgnn.registry"
paper_title: "Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting"
venue: "AAAI 2024"
year: 2024
arxiv: ""
---
# STPGNN

STPGNN (Spatio-Temporal Pivotal Graph Neural Network) is a spatiotemporal learning model for node-structured traffic forecasting that explicitly identifies and models pivotal nodes — nodes with a large number of connections to other nodes — which are disproportionately difficult to predict with standard graph neural networks. It consists of a Pivotal Node Identification Module, a Pivotal Graph Convolution Module for capturing complex spatio-temporal dependencies around these high-connectivity nodes, and a parallel architecture that simultaneously processes both pivotal and non-pivotal nodes.

## Paper
- **Title**: Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting
- **Venue**: AAAI 2024
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Traffic flow forecasting is a classical spatio-temporal data mining problem with many real-world applications. Graph Neural Networks (GNNs) are currently the mainstream approach to solving this problem. However, the majority of existing methods disregard the importance of certain nodes (referred to as pivotal nodes) that naturally exhibit extensive connections with multiple other nodes. Predicting on pivotal nodes poses a challenge due to their complex spatio-temporal dependencies compared to other nodes. In this paper, we propose Spatio-Temporal Pivotal Graph Neural Networks (STPGNN) to address this challenge. Specifically, we first introduce a pivotal node identification module for identifying pivotal nodes. We then propose a novel pivotal graph convolution module, enabling precise capture of spatio-temporal dependencies centered around pivotal nodes. We further propose a parallel framework capable of extracting spatio-temporal traffic features on both pivotal and non-pivotal nodes. Experiments on seven real-world traffic datasets verify the effectiveness and efficiency of our proposed method compared to state-of-the-art baselines.

## In ModernTSF
Default config: `configs/models/STPGNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
