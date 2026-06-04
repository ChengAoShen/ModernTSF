---
model: "DGCRN"
forecasting_setting: "spatiotemporal"
config: "configs/models/DGCRN.toml"
registry: "models.dgcrn.registry"
paper_title: "Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution"
venue: "ACM TKDD 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2104.14917"
---
# DGCRN

DGCRN (Dynamic Graph Convolutional Recurrent Network) is a spatiotemporal model for traffic forecasting that captures time-varying node correlations on road networks. At each time step, hyper-networks generate dynamic filter parameters and produce a dynamic adjacency matrix that is integrated with a pre-defined static graph inside a GRU-style recurrent cell, enabling fine-grained modeling of evolving spatial dependencies.

## Paper
- **Title**: Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution
- **Venue**: ACM Transactions on Knowledge Discovery from Data (TKDD), Vol. 17, No. 1, Article 9
- **Published**: 2023 (arXiv: 2021-04)
- **arXiv**: https://arxiv.org/abs/2104.14917

## Abstract
Traffic prediction is the cornerstone of an intelligent transportation system. Accurate traffic forecasting is essential for the applications of smart cities, i.e., intelligent traffic management and urban planning. Although various methods are proposed for spatio-temporal modeling, they ignore the dynamic characteristics of correlations among locations on road networks. Meanwhile, most Recurrent Neural Network (RNN) based works are not efficient enough due to their recurrent operations. Additionally, there is a severe lack of fair comparison among different methods on the same datasets. To address the above challenges, in this paper, we propose a novel traffic prediction framework, named Dynamic Graph Convolutional Recurrent Network (DGCRN). In DGCRN, hyper-networks are designed to leverage and extract dynamic characteristics from node attributes, while the parameters of dynamic filters are generated at each time step. We filter the node embeddings and then use them to generate a dynamic graph, which is integrated with a pre-defined static graph. As far as we know, we are the first to employ a generation method to model fine topology of dynamic graph at each time step. Further, to enhance efficiency and performance, we employ a training strategy for DGCRN by restricting the iteration number of decoder during forward and backward propagation. Finally, a reproducible standardized benchmark and a brand new representative traffic dataset are opened for fair comparison and further research. Extensive experiments on three datasets demonstrate that our model outperforms 15 baselines consistently.

## In ModernTSF
Default config: `configs/models/DGCRN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
