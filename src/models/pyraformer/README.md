---
model: "Pyraformer"
forecasting_setting: "time_series"
config: "configs/models/Pyraformer.toml"
registry: "models.pyraformer.registry"
paper_title: "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting"
venue: "ICLR 2022"
year: 2022
arxiv: ""
---
# Pyraformer

Pyraformer is a Transformer-based time series forecasting model that builds a multi-resolution pyramidal attention module (PAM) over the input sequence. Inter-scale tree connections summarize temporal features at progressively coarser resolutions, while intra-scale connections between neighboring tokens model dependencies at each resolution. This design achieves O(1) maximum signal-path length with respect to sequence length and linear time and space complexity, making it efficient for long-range forecasting on both single-step and multi-step horizons.

## Paper
- **Title**: Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting
- **Venue**: ICLR 2022 (Oral)
- **Published**: 2022
- **arXiv**: N/A

## Abstract
Accurate prediction of the future given the past based on time series data is of paramount importance, since it opens the door for decision making and risk management ahead of time. In practice, the challenge is to build a flexible but parsimonious model that can capture a wide range of temporal dependencies. In this paper, we propose Pyraformer by exploring the multiresolution representation of the time series. Specifically, we introduce the pyramidal attention module (PAM) in which the inter-scale tree structure summarizes features at different resolutions and the intra-scale neighboring connections model the temporal dependencies of different ranges. Under mild conditions, the maximum length of the signal traversing path in Pyraformer is a constant (i.e., O(1)) with regard to the sequence length L, while its time and space complexity scale linearly with L. Extensive numerical results show that Pyraformer typically achieves the highest prediction accuracy in both single-step and long-range forecasting tasks with the least amount of time and memory consumption, especially when the sequence is long.

## In ModernTSF
Default config: `configs/models/Pyraformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LiuYLLLLD22,
  author       = {Shizhan Liu and
                  Hang Yu and
                  Cong Liao and
                  Jianguo Li and
                  Weiyao Lin and
                  Alex X. Liu and
                  Schahram Dustdar},
  title        = {Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time
                  Series Modeling and Forecasting},
  booktitle    = {The Tenth International Conference on Learning Representations, {ICLR}
                  2022, Virtual Event, April 25-29, 2022},
  publisher    = {OpenReview.net},
  year         = {2022},
  url          = {https://openreview.net/forum?id=0EXmFzUn5I},
  code         = {https://github.com/ant-research/Pyraformer},
  timestamp    = {Fri, 02 Aug 2024 21:59:25 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/LiuYLLLLD22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
