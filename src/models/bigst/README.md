---
model: "BigST"
category: "spatiotemporal_learning"
category_name: "Spatiotemporal Learning"
forecasting_setting: "spatiotemporal"
config: "configs/models/BigST.toml"
registry: "models.bigst.registry"
paper_title: "BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks"
venue: "PVLDB 2024"
year: 2024
arxiv: ""
---
# BigST

BigST is a spatiotemporal learning model designed for large-scale traffic forecasting on road networks. It models both temporal dynamics and spatial dependencies among nodes, scaling to graphs with up to one hundred thousand nodes by replacing the conventional quadratic-complexity graph attention with a linearized random-feature approximation and a pre-computable long-range temporal encoder.

## Paper
- **Title**: BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks
- **Venue**: Proceedings of the VLDB Endowment (PVLDB), Vol. 17, No. 5, pp. 1081–1090
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Spatio-Temporal Graph Neural Network (STGNN) has been used as a common workhorse for traffic forecasting. However, most of them require prohibitive quadratic computational complexity to capture long-range spatio-temporal dependencies, thus hindering their applications to long historical sequences on large-scale road networks in the real-world. To this end, in this paper, we propose BigST, a linear complexity spatio-temporal graph neural network, to efficiently exploit long-range spatio-temporal dependencies for large-scale traffic forecasting. Specifically, we first propose a scalable long sequence feature extractor to encode node-wise longrange inputs (e.g., thousands of time-steps in the past week) into low-dimensional representations encompassing rich temporal dynamics. The resulting representations can be pre-computed and hence significantly reduce the computational overhead for prediction. Then, we build a linearized global spatial convolution network to adaptively distill time-varying graph structures, which enables fast runtime message passing along spatial dimensions in linear complexity. We empirically evaluate our model on two large-scale real-world traffic datasets. Extensive experiments demonstrate that BigST can scale to road networks with up to one hundred thousand nodes, while significantly improving prediction accuracy and efficiency compared to state-of-the-art traffic forecasting models.

## In ModernTSF
Default config: `configs/models/BigST.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
