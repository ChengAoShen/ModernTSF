---
model: "PM25_GNN"
category: "covariate_prediction"
category_name: "Covariate Prediction"
forecasting_setting: "covariate"
config: "configs/models/PM25_GNN.toml"
registry: "models.pm25gnn.registry"
paper_title: "PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting"
venue: "ACM SIGSPATIAL 2020"
year: 2020
arxiv: "https://arxiv.org/abs/2002.12898"
---
# PM25_GNN

PM25_GNN is a graph neural network model for air quality (PM2.5 concentration) forecasting that integrates domain knowledge about pollutant diffusion processes to construct the graph topology and combines GNN layers with GRU-based temporal modeling to capture both fine-grained and long-term spatial-temporal dependencies across monitoring stations.

## Paper
- **Title**: PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting
- **Venue**: ACM SIGSPATIAL 2020
- **Published**: 2020 (arXiv: 2020-02)
- **arXiv**: https://arxiv.org/abs/2002.12898

## Abstract
When predicting PM2.5 concentrations, it is necessary to consider complex information sources since the concentrations are influenced by various factors within a long period. In this paper, we identify a set of critical domain knowledge for PM2.5 forecasting and develop a novel graph based model, PM2.5-GNN, being capable of capturing long-term dependencies. On a real-world dataset, we validate the effectiveness of the proposed model and examine its abilities of capturing both fine-grained and long-term influences in PM2.5 process. The proposed PM2.5-GNN has also been deployed online to provide free forecasting service.

## In ModernTSF
Default config: `configs/models/PM25_GNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
