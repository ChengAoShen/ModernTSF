---
model: "GTS"
forecasting_setting: "spatiotemporal"
config: "configs/models/GTS.toml"
registry: "models.gts.registry"
paper_title: "Discrete Graph Structure Learning for Forecasting Multiple Time Series"
venue: "ICLR 2021"
year: 2021
arxiv: "https://arxiv.org/abs/2101.06861"
---
# GTS

GTS (Graph for Time Series) is a spatiotemporal learning model that jointly learns a discrete probabilistic graph structure and a DCRNN-style graph convolutional recurrent forecaster from multivariate time series data. Rather than relying on a pre-defined adjacency matrix, GTS parameterises the graph distribution with a neural network and samples discrete graphs differentiably via reparameterisation, so that the graph topology and the forecasting model are optimised end-to-end.

## Paper
- **Title**: Discrete Graph Structure Learning for Forecasting Multiple Time Series
- **Venue**: ICLR 2021
- **Published**: 2021 (arXiv: 2021-01)
- **arXiv**: https://arxiv.org/abs/2101.06861

## Abstract
Time series forecasting is an extensively studied subject in statistics, economics, and computer science. Exploration of the correlation and causation among the variables in a multivariate time series shows promise in enhancing the performance of a time series model. When using deep neural networks as forecasting models, we hypothesize that exploiting the pairwise information among multiple (multivariate) time series also improves their forecast. If an explicit graph structure is known, graph neural networks (GNNs) have been demonstrated as powerful tools to exploit the structure. In this work, we propose learning the structure simultaneously with the GNN if the graph is unknown. We cast the problem as learning a probabilistic graph model through optimizing the mean performance over the graph distribution. The distribution is parameterized by a neural network so that discrete graphs can be sampled differentiably through reparameterization. Empirical evaluations show that our method is simpler, more efficient, and better performing than a recently proposed bilevel learning approach for graph structure learning, as well as a broad array of forecasting models, either deep or non-deep learning based, and graph or non-graph based.

## In ModernTSF
Default config: `configs/models/GTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/Shang0B21,
  author       = {Chao Shang and
                  Jie Chen and
                  Jinbo Bi},
  title        = {Discrete Graph Structure Learning for Forecasting Multiple Time Series},
  booktitle    = {9th International Conference on Learning Representations, {ICLR} 2021,
                  Virtual Event, Austria, May 3-7, 2021},
  publisher    = {OpenReview.net},
  year         = {2021},
  url          = {https://openreview.net/forum?id=WEHSlH5mOk},
  timestamp    = {Wed, 23 Jun 2021 17:36:39 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/Shang0B21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
