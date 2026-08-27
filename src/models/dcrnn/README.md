---
name: "DCRNN"
implementation: rewrite
summary: "The DCRNN paper combines bidirectional random-walk diffusion convolution with a recurrent encoder-decoder and scheduled sampling for multi-step graph traffic forecasting. This ModernTSF entry is a secondary PyTorch implementation that retains the dual-random-walk recurrent architecture, but its default contract disables scheduled sampling and does not reproduce the official TensorFlow experiment pipeline."
paper:
  title: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"
  venue: "ICLR 2018"
  year: 2018
  url: "https://openreview.net/forum?id=SJiHXGWAZ"
codebase:
  url: "https://github.com/liyaguang/DCRNN"
  revision: "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
  license: "MIT"
  usage: reference-only
---
# DCRNN

The DCRNN paper combines bidirectional random-walk diffusion convolution with a recurrent encoder-decoder and scheduled sampling for multi-step graph traffic forecasting. This ModernTSF entry is a secondary PyTorch implementation that retains the dual-random-walk recurrent architecture, but its default contract disables scheduled sampling and does not reproduce the official TensorFlow experiment pipeline.

<!-- model-card:canonical:start -->
## Method overview

The DCRNN paper combines bidirectional random-walk diffusion convolution with a recurrent encoder-decoder and scheduled sampling for multi-step graph traffic forecasting.

## Core architecture

This ModernTSF entry is a secondary PyTorch implementation that retains the dual-random-walk recurrent architecture, but its default contract disables scheduled sampling and does not reproduce the official TensorFlow experiment pipeline.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Graph adjacency is supplied at construction; temporal/node covariates follow the runtime batch contract.

## Paper and code

- [paper](https://openreview.net/forum?id=SJiHXGWAZ); title: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting; venue/year: ICLR 2018 / 2018
- [codebase](https://github.com/liyaguang/DCRNN); revision: `602afd9d767d3aa1c9b3eac51710d6aeee12c227`; license: `MIT`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/DCRNN.toml`](../../../configs/models/DCRNN.toml).

## Differences

- Official source: https://github.com/liyaguang/DCRNN at `602afd9d767d3aa1c9b3eac51710d6aeee12c227` (MIT).
Implementation: **rewrite** (clean-room audit pending). The local PyTorch implementation was adapted from BasicTS, but its exact BasicTS source revision was not recorded and no numerical comparison with the official TensorFlow code exists.
- Known differences: the preset uses one 16-unit recurrent layer and three input channels versus the official METR-LA preset's two 64-unit layers and two input channels. Scheduled sampling is disabled, missing adjacency falls back to identity supports, and official preprocessing and masked-MAE training are outside the model package.

## Shared components

- [`graph_utils`](../../components/graph_utils.py)
- [`marks`](../../components/marks.py)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `rnn_units=16`, `num_rnn_layers=1`, `max_diffusion_step=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
- **Venue**: ICLR 2018
- **Published**: 2018 (arXiv: 2017-07)
- **arXiv**: https://arxiv.org/abs/1707.01926

## Abstract
Spatiotemporal forecasting has various applications in neuroscience, climate and transportation domain. Traffic forecasting is one canonical example of such learning task. The task is challenging due to (1) complex spatial dependency on road networks, (2) non-linear temporal dynamics with changing road conditions and (3) inherent difficulty of long-term forecasting. To address these challenges, we propose to model the traffic flow as a diffusion process on a directed graph and introduce Diffusion Convolutional Recurrent Neural Network (DCRNN), a deep learning framework for traffic forecasting that incorporates both spatial and temporal dependency in the traffic flow. Specifically, DCRNN captures the spatial dependency using bidirectional random walks on the graph, and the temporal dependency using the encoder-decoder architecture with scheduled sampling. We evaluate the framework on two real-world large scale road network traffic datasets and observe consistent improvement of 12% - 15% over state-of-the-art baselines.

## In ModernTSF
Default config: `configs/models/DCRNN.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

- Official source: https://github.com/liyaguang/DCRNN at `602afd9d767d3aa1c9b3eac51710d6aeee12c227` (MIT).
Implementation: **rewrite** (clean-room audit pending). The local PyTorch implementation was adapted from BasicTS, but its exact BasicTS source revision was not recorded and no numerical comparison with the official TensorFlow code exists.
- Known differences: the preset uses one 16-unit recurrent layer and three input channels versus the official METR-LA preset's two 64-unit layers and two input channels. Scheduled sampling is disabled, missing adjacency falls back to identity supports, and official preprocessing and masked-MAE training are outside the model package.

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
