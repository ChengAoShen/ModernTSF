---
model: "DSTAGNN"
forecasting_setting: "spatiotemporal"
config: "configs/models/DSTAGNN.toml"
registry: "models.dstagnn.registry"
paper_title: "DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting"
venue: "ICML 2022"
year: 2022
arxiv: ""
---
# DSTAGNN

DSTAGNN is a spatiotemporal learning model for node-structured and graph-structured data. It simultaneously models temporal dependencies and dynamic spatial relationships among nodes in a road network, producing forecasts for all nodes' future target values using a data-driven dynamic graph and multi-scale gated convolution.

## Paper
- **Title**: DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting
- **Venue**: ICML 2022
- **Published**: 2022
- **arXiv**: N/A

## Abstract
As a typical problem in time series analysis, traffic flow prediction is one of the most important application fields of machine learning. However, achieving highly accurate traffic flow prediction is a challenging task, due to the presence of complex dynamic spatial-temporal dependencies within a road network. This paper proposes a novel Dynamic Spatial-Temporal Aware Graph Neural Network (DSTAGNN) to model the complex spatial-temporal interaction in road network. First, considering the fact that historical data carries intrinsic dynamic information about the spatial structure of road networks, we propose a new dynamic spatial-temporal aware graph based on a data-driven strategy to replace the pre-defined static graph usually used in traditional graph convolution. Second, we design a novel graph neural network architecture, which can not only represent dynamic spatial relevance among nodes with an improved multi-head attention mechanism, but also acquire the wide range of dynamic temporal dependency from multi-receptive field features via multi-scale gated convolution. Extensive experiments on real-world data sets demonstrate that our proposed method significantly outperforms the state-of-the-art methods.

## In ModernTSF
Default config: `configs/models/DSTAGNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/LanMHWYL22,
  author       = {Shiyong Lan and
                  Yitong Ma and
                  Weikang Huang and
                  Wenwu Wang and
                  Hongyu Yang and
                  Pyang Li},
  editor       = {Kamalika Chaudhuri and
                  Stefanie Jegelka and
                  Le Song and
                  Csaba Szepesv{\'{a}}ri and
                  Gang Niu and
                  Sivan Sabato},
  title        = {{DSTAGNN:} Dynamic Spatial-Temporal Aware Graph Neural Network for
                  Traffic Flow Forecasting},
  booktitle    = {International Conference on Machine Learning, {ICML} 2022, 17-23 July
                  2022, Baltimore, Maryland, {USA}},
  series       = {Proceedings of Machine Learning Research},
  pages        = {11906--11917},
  publisher    = {{PMLR}},
  year         = {2022},
  url          = {https://proceedings.mlr.press/v162/lan22a.html},
  timestamp    = {Thu, 05 Jan 2023 08:20:54 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/LanMHWYL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
