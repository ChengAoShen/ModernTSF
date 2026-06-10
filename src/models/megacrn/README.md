---
model: "MegaCRN"
forecasting_setting: "spatiotemporal"
config: "configs/models/MegaCRN.toml"
registry: "models.megacrn.registry"
paper_title: "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting"
venue: "AAAI 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2211.14701"
---
# MegaCRN

MegaCRN (Meta-Graph Convolutional Recurrent Network) is a spatiotemporal forecasting model designed for graph-structured node data such as road-network traffic. It addresses the heterogeneity and non-stationarity inherent in traffic streams by learning dynamic graph structures through a Meta-Graph Learner backed by a learnable Meta-Node Bank, plugged into a GCRN encoder-decoder. This allows the model to disentangle locations and time slots with different patterns and adapt robustly to anomalous conditions.

## Paper
- **Title**: Spatio-Temporal Meta-Graph Learning for Traffic Forecasting
- **Venue**: AAAI 2023
- **Published**: 2023 (arXiv: 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.14701

## Abstract
Traffic forecasting as a canonical task of multivariate time series forecasting has been a significant research topic in AI community. To address the spatio-temporal heterogeneity and non-stationarity implied in the traffic stream, in this study, we propose Spatio-Temporal Meta-Graph Learning as a novel Graph Structure Learning mechanism on spatio-temporal data. Specifically, we implement this idea into Meta-Graph Convolutional Recurrent Network (MegaCRN) by plugging the Meta-Graph Learner powered by a Meta-Node Bank into GCRN encoder-decoder. We conduct a comprehensive evaluation on two benchmark datasets (i.e., METR-LA and PEMS-BAY) and a new large-scale traffic speed dataset called EXPY-TKY that covers 1843 expressway road links in Tokyo. Our model outperformed the state-of-the-arts on all three datasets. Besides, through a series of qualitative evaluations, we demonstrate that our model can explicitly disentangle the road links and time slots with different patterns and be robustly adaptive to any anomalous traffic situations.

## In ModernTSF
Default config: `configs/models/MegaCRN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/Jiang0YJCK0FS23,
  author       = {Renhe Jiang and
                  Zhaonan Wang and
                  Jiawei Yong and
                  Puneet Jeph and
                  Quanjun Chen and
                  Yasumasa Kobayashi and
                  Xuan Song and
                  Shintaro Fukushima and
                  Toyotaro Suzumura},
  editor       = {Brian Williams and
                  Yiling Chen and
                  Jennifer Neville},
  title        = {Spatio-Temporal Meta-Graph Learning for Traffic Forecasting},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                  7-14, 2023},
  pages        = {8078--8086},
  publisher    = {{AAAI} Press},
  year         = {2023},
  url          = {https://doi.org/10.1609/aaai.v37i7.25976},
  doi          = {10.1609/AAAI.V37I7.25976},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/Jiang0YJCK0FS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
