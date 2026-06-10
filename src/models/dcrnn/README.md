---
model: "DCRNN"
forecasting_setting: "spatiotemporal"
config: "configs/models/DCRNN.toml"
registry: "models.dcrnn.registry"
paper_title: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"
venue: "ICLR 2018"
year: 2018
arxiv: "https://arxiv.org/abs/1707.01926"
---
# DCRNN

DCRNN (Diffusion Convolutional Recurrent Neural Network) is a spatiotemporal learning model for node-structured graph data, designed for traffic flow forecasting over road networks. It models traffic flow as a diffusion process on a directed graph, capturing spatial dependencies via bidirectional random walks and temporal dependencies using an encoder-decoder architecture with scheduled sampling, enabling accurate multi-step ahead forecasting across all graph nodes.

## Paper
- **Title**: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
- **Venue**: ICLR 2018
- **Published**: 2018 (arXiv: 2017-07)
- **arXiv**: https://arxiv.org/abs/1707.01926

## Abstract
Spatiotemporal forecasting has various applications in neuroscience, climate and transportation domain. Traffic forecasting is one canonical example of such learning task. The task is challenging due to (1) complex spatial dependency on road networks, (2) non-linear temporal dynamics with changing road conditions and (3) inherent difficulty of long-term forecasting. To address these challenges, we propose to model the traffic flow as a diffusion process on a directed graph and introduce Diffusion Convolutional Recurrent Neural Network (DCRNN), a deep learning framework for traffic forecasting that incorporates both spatial and temporal dependency in the traffic flow. Specifically, DCRNN captures the spatial dependency using bidirectional random walks on the graph, and the temporal dependency using the encoder-decoder architecture with scheduled sampling. We evaluate the framework on two real-world large scale road network traffic datasets and observe consistent improvement of 12% - 15% over state-of-the-art baselines.

## In ModernTSF
Default config: `configs/models/DCRNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LiYS018,
  author       = {Yaguang Li and
                  Rose Yu and
                  Cyrus Shahabi and
                  Yan Liu},
  title        = {Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
                  Forecasting},
  booktitle    = {6th International Conference on Learning Representations, {ICLR} 2018,
                  Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
  publisher    = {OpenReview.net},
  year         = {2018},
  url          = {https://openreview.net/forum?id=SJiHXGWAZ},
  timestamp    = {Thu, 25 Jul 2019 14:25:46 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/LiYS018.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
