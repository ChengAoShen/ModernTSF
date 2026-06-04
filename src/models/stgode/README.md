---
model: "STGODE"
forecasting_setting: "spatiotemporal"
config: "configs/models/STGODE.toml"
registry: "models.stgode.registry"
paper_title: "Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting"
venue: "KDD 2021"
year: 2021
arxiv: "https://arxiv.org/abs/2106.12931"
---
# STGODE

STGODE is a spatiotemporal learning model for node-structured traffic and graph data that captures continuous spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE). By coupling a semantic adjacency matrix with a temporal dilated convolution structure, it overcomes the over-smoothing limitation of shallow GNNs and captures both structural and semantic long-range dependencies between nodes.

## Paper
- **Title**: Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting
- **Venue**: KDD 2021
- **Published**: 2021 (arXiv: 2021-06)
- **arXiv**: https://arxiv.org/abs/2106.12931

## Abstract
Spatial-temporal forecasting has attracted tremendous attention in a wide range of applications, and traffic flow prediction is a canonical and typical example. The complex and long-range spatial-temporal correlations of traffic flow bring it to a most intractable challenge. Existing works typically utilize shallow graph convolution networks (GNNs) and temporal extracting modules to model spatial and temporal dependencies respectively. However, the representation ability of such models is limited due to: (1) shallow GNNs are incapable to capture long-range spatial correlations, (2) only spatial connections are considered and a mass of semantic connections are ignored, which are of great importance for a comprehensive understanding of traffic networks. To this end, we propose Spatial-Temporal Graph Ordinary Differential Equation Networks (STGODE). Specifically, we capture spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE), as a result, deeper networks can be constructed and spatial-temporal features are utilized synchronously. To understand the network more comprehensively, semantical adjacency matrix is considered in our model, and a well-design temporal dialated convolution structure is used to capture long term temporal dependencies. We evaluate our model on multiple real-world traffic datasets and superior performance is achieved over state-of-the-art baselines.

## In ModernTSF
Default config: `configs/models/STGODE.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
