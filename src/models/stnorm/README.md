---
model: "STNorm"
forecasting_setting: "spatiotemporal"
config: "configs/models/STNorm.toml"
registry: "models.stnorm.registry"
paper_title: "ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting"
venue: "KDD 2021"
year: 2021
arxiv: ""
---
# STNorm

STNorm is a spatiotemporal forecasting model that augments a WaveNet-style backbone with two dedicated normalization modules — spatial normalization and temporal normalization — to separately refine high-frequency temporal components and local spatial components in multi-variate time-series data. It operates on node-structured data and does not require an externally provided static adjacency matrix.

## Paper
- **Title**: ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting
- **Venue**: KDD 2021
- **Published**: 2021
- **arXiv**: N/A

## Abstract
Multi-variate time series (MTS) data is generated from hybrid dynamical systems with unknown dynamics. The hybrid nature of such systems is a result of complex external impacts, which can be summarized as high-frequency and low-frequency from the temporal view, or global and local if we take the spatial view. These impacts are paramount to capture in time series forecasting tasks. In this paper, we propose temporal and spatial normalization modules which separately refine the high-frequency component and the local component underlying the raw data and can be integrated into canonical deep learning architectures such as WaveNet and Transformer. We conduct extensive experiments to demonstrate that the proposed method achieves superior performance on two public traffic network datasets, METR-LA and PEMS-BAY.

## In ModernTSF
Default config: `configs/models/STNorm.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/DengCJST21,
  author       = {Jinliang Deng and
                  Xiusi Chen and
                  Renhe Jiang and
                  Xuan Song and
                  Ivor W. Tsang},
  editor       = {Feida Zhu and
                  Beng Chin Ooi and
                  Chunyan Miao},
  title        = {ST-Norm: Spatial and Temporal Normalization for Multi-variate Time
                  Series Forecasting},
  booktitle    = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
  pages        = {269--278},
  publisher    = {{ACM}},
  year         = {2021},
  url          = {https://doi.org/10.1145/3447548.3467330},
  doi          = {10.1145/3447548.3467330},
  timestamp    = {Tue, 07 May 2024 20:08:07 +0200},
  biburl       = {https://dblp.org/rec/conf/kdd/DengCJST21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
