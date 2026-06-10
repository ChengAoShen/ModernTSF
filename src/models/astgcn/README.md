---
model: "ASTGCN"
forecasting_setting: "covariate"
config: "configs/models/ASTGCN.toml"
registry: "models.astgcn.registry"
paper_title: "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting"
venue: "AAAI 2019"
year: 2019
arxiv: ""
---
# ASTGCN

ASTGCN (Attention Based Spatial-Temporal Graph Convolutional Network) is a covariate prediction model designed for node-structured spatial-temporal data, such as traffic flow forecasting on road networks. It captures dynamic spatial-temporal correlations by combining spatial-temporal attention mechanisms with graph convolutions (Chebyshev-basis) for spatial patterns and standard convolutions for temporal features, processing three independent temporal components (recent, daily-periodic, weekly-periodic) whose outputs are weighted-fused to produce final predictions.

## Paper
- **Title**: Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting
- **Venue**: AAAI 2019
- **Published**: 2019
- **arXiv**: N/A

## Abstract
Forecasting the traffic flows is a critical issue for researchers and practitioners in the field of transportation. However, it is very challenging since the traffic flows usually show high nonlinearities and complex patterns. Most existing traffic flow prediction methods, lacking abilities of modeling the dynamic spatial-temporal correlations of traffic data, thus cannot yield satisfactory prediction results. In this paper, we propose a novel attention based spatial-temporal graph convolutional network (ASTGCN) model to solve traffic flow forecasting problem. ASTGCN mainly consists of three independent components to respectively model three temporal properties of traffic flows, i.e., recent, daily-periodic and weekly-periodic dependencies. More specifically, each component contains two major parts: 1) the spatial-temporal attention mechanism to effectively capture the dynamic spatialtemporal correlations in traffic data; 2) the spatial-temporal convolution which simultaneously employs graph convolutions to capture the spatial patterns and common standard convolutions to describe the temporal features. The output of the three components are weighted fused to generate the final prediction results. Experiments on two real-world datasets from the Caltrans Performance Measurement System (PeMS) demonstrate that the proposed ASTGCN model outperforms the state-of-the-art baselines.

## In ModernTSF
Default config: `configs/models/ASTGCN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/GuoLFSW19,
  author       = {Shengnan Guo and
                  Youfang Lin and
                  Ning Feng and
                  Chao Song and
                  Huaiyu Wan},
  title        = {Attention Based Spatial-Temporal Graph Convolutional Networks for
                  Traffic Flow Forecasting},
  booktitle    = {The Thirty-Third {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2019, The Thirty-First Innovative Applications of Artificial Intelligence
                  Conference, {IAAI} 2019, The Ninth {AAAI} Symposium on Educational
                  Advances in Artificial Intelligence, {EAAI} 2019, Honolulu, Hawaii,
                  USA, January 27 - February 1, 2019},
  pages        = {922--929},
  publisher    = {{AAAI} Press},
  year         = {2019},
  url          = {https://doi.org/10.1609/aaai.v33i01.3301922},
  doi          = {10.1609/AAAI.V33I01.3301922},
  timestamp    = {Mon, 04 Sep 2023 12:29:24 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/GuoLFSW19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
