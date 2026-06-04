---
model: "STGCN"
forecasting_setting: "spatiotemporal"
config: "configs/models/STGCN.toml"
registry: "models.stgcn.registry"
paper_title: "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"
venue: "IJCAI 2018"
year: 2018
arxiv: "https://arxiv.org/abs/1709.04875"
---
# STGCN

STGCN (Spatio-Temporal Graph Convolutional Network) is a deep learning framework for node-level spatiotemporal forecasting, originally developed for traffic speed prediction. It combines graph convolution layers that capture spatial dependencies between nodes on a road network with temporal convolution layers that model short- and long-range time patterns, using fully convolutional structures to achieve fast training and compact parameterisation compared to recurrent alternatives.

## Paper
- **Title**: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
- **Venue**: IJCAI 2018
- **Published**: 2018 (arXiv: 2017-09)
- **arXiv**: https://arxiv.org/abs/1709.04875

## Abstract
Timely accurate traffic forecast is crucial for urban traffic control and guidance. Due to the high nonlinearity and complexity of traffic flow, traditional methods cannot satisfy the requirements of mid-and-long term prediction tasks and often neglect spatial and temporal dependencies. In this paper, we propose a novel deep learning framework, Spatio-Temporal Graph Convolutional Networks (STGCN), to tackle the time series prediction problem in traffic domain. Instead of applying regular convolutional and recurrent units, we formulate the problem on graphs and build the model with complete convolutional structures, which enable much faster training speed with fewer parameters. Experiments show that our model STGCN effectively captures comprehensive spatio-temporal correlations through modeling multi-scale traffic networks and consistently outperforms state-of-the-art baselines on various real-world traffic datasets.

## In ModernTSF
Default config: `configs/models/STGCN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
