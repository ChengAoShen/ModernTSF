---
model: "GAGNN"
forecasting_setting: "covariate"
config: "configs/models/GAGNN.toml"
registry: "models.gagnn.registry"
paper_title: "Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting"
venue: "ACM TKDD 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2108.12238"
---
# GAGNN

GAGNN is a covariate prediction model for node-level air quality forecasting, corresponding to the original air quality prediction setting. It constructs both a city graph and a city group graph to capture spatial and latent dependencies between cities, using hierarchical group-aware attention and message-passing to predict future air quality indices at each node.

## Paper
- **Title**: Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting
- **Venue**: ACM Transactions on Knowledge Discovery from Data (TKDD), Vol. 18, No. 3, Article 55
- **Published**: 2024 (arXiv: 2021-08)
- **arXiv**: https://arxiv.org/abs/2108.12238

## Abstract
The problem of air pollution threatens public health. Air quality forecasting can provide the air quality index hours or even days later, which can help the public to prevent air pollution in advance. Previous works focus on citywide air quality forecasting and cannot solve nationwide city forecasting problem, whose difficulties lie in capturing the latent dependencies between geographically distant but highly correlated cities. In this paper, we propose the group-aware graph neural network (GAGNN), a hierarchical model for nationwide city air quality forecasting. The model constructs a city graph and a city group graph to model the spatial and latent dependencies between cities, respectively. GAGNN introduces differentiable grouping network to discover the latent dependencies among cities and generate city groups. Based on the generated city groups, a group correlation encoding module is introduced to learn the correlations between them, which can effectively capture the dependencies between city groups. After the graph construction, GAGNN implements message passing mechanism to model the dependencies between cities and city groups. The evaluation experiments on Chinese city air quality dataset indicate that our GAGNN outperforms existing forecasting models.

## In ModernTSF
Default config: `configs/models/GAGNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/tkdd/ChenXWH24,
  author       = {Ling Chen and
                  Jiahui Xu and
                  Binqing Wu and
                  Jianlong Huang},
  title        = {Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting},
  journal      = {{ACM} Trans. Knowl. Discov. Data},
  volume       = {18},
  number       = {3},
  pages        = {55:1--55:20},
  year         = {2024},
  url          = {https://doi.org/10.1145/3631713},
  doi          = {10.1145/3631713},
  timestamp    = {Sun, 19 Jan 2025 14:58:36 +0100},
  biburl       = {https://dblp.org/rec/journals/tkdd/ChenXWH24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
