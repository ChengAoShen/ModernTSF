---
model: "D2STGNN"
forecasting_setting: "spatiotemporal"
config: "configs/models/D2STGNN.toml"
registry: "models.d2stgnn.registry"
paper_title: "Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting"
venue: "VLDB 2022"
year: 2022
arxiv: "https://arxiv.org/abs/2206.09112"
---
# D2STGNN

D2STGNN (Decoupled Dynamic Spatial-Temporal Graph Neural Network) is a spatiotemporal learning model designed for node-structured graph data such as road-sensor traffic networks. It explicitly separates traffic signals into diffusion signals (vehicles propagating through the network) and inherent signals (local non-diffusion patterns) via a learned estimation gate and residual decomposition, then processes each component with a dedicated module while a dynamic graph learning sub-network captures time-varying spatial topology.

## Paper
- **Title**: Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting
- **Venue**: VLDB 2022
- **Published**: 2022 (arXiv: 2022-06)
- **arXiv**: https://arxiv.org/abs/2206.09112

## Abstract
We all depend on mobility, and vehicular transportation affects the daily lives of most of us. Thus, the ability to forecast the state of traffic in a road network is an important functionality and a challenging task. Traffic data is often obtained from sensors deployed in a road network. Recent proposals on spatial-temporal graph neural networks have achieved great progress at modeling complex spatial-temporal correlations in traffic data, by modeling traffic data as a diffusion process. However, intuitively, traffic data encompasses two different kinds of hidden time series signals, namely the diffusion signals and inherent signals. Unfortunately, nearly all previous works coarsely consider traffic signals entirely as the outcome of the diffusion, while neglecting the inherent signals, which impacts model performance negatively. To improve modeling performance, we propose a novel Decoupled Spatial-Temporal Framework (DSTF) that separates the diffusion and inherent traffic information in a data-driven manner, which encompasses a unique estimation gate and a residual decomposition mechanism. The separated signals can be handled subsequently by the diffusion and inherent modules separately. Further, we propose an instantiation of DSTF, Decoupled Dynamic Spatial-Temporal Graph Neural Network (D2STGNN), that captures spatial-temporal correlations and also features a dynamic graph learning module that targets the learning of the dynamic characteristics of traffic networks. Extensive experiments with four real-world traffic datasets demonstrate that the framework is capable of advancing the state-of-the-art.

## In ModernTSF
Default config: `configs/models/D2STGNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/pvldb/ShaoZWWXCJ22,
  author       = {Zezhi Shao and
                  Zhao Zhang and
                  Wei Wei and
                  Fei Wang and
                  Yongjun Xu and
                  Xin Cao and
                  Christian S. Jensen},
  title        = {Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic
                  Forecasting},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {15},
  number       = {11},
  pages        = {2733--2746},
  year         = {2022},
  url          = {https://www.vldb.org/pvldb/vol15/p2733-shao.pdf},
  doi          = {10.14778/3551793.3551827},
  timestamp    = {Sat, 06 Sep 2025 20:28:21 +0200},
  biburl       = {https://dblp.org/rec/journals/pvldb/ShaoZWWXCJ22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
