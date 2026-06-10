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

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/KongGL24,
  author       = {Weiyang Kong and
                  Ziyu Guo and
                  Yubao Liu},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {8627--8635},
  publisher    = {{AAAI} Press},
  year         = {2024},
  url          = {https://doi.org/10.1609/aaai.v38i8.28707},
  doi          = {10.1609/AAAI.V38I8.28707},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/KongGL24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
