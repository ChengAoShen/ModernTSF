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

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/0001YL0020,
  author       = {Lei Bai and
                  Lina Yao and
                  Can Li and
                  Xianzhi Wang and
                  Can Wang},
  editor       = {Hugo Larochelle and
                  Marc'Aurelio Ranzato and
                  Raia Hadsell and
                  Maria{-}Florina Balcan and
                  Hsuan{-}Tien Lin},
  title        = {Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 33: Annual Conference
                  on Neural Information Processing Systems 2020, NeurIPS 2020, December
                  6-12, 2020, virtual},
  year         = {2020},
  url          = {https://proceedings.neurips.cc/paper/2020/hash/ce1aad92b939420fc17005e5461e6f48-Abstract.html},
  timestamp    = {Sun, 02 Nov 2025 10:11:42 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/0001YL0020.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
