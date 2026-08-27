---
name: "ASTGCN"
implementation: rewrite
summary: "The ASTGCN paper proposes a graph traffic forecaster with spatial-temporal attention, Chebyshev graph convolution, temporal convolution, and a learned fusion of recent, daily-periodic, and weekly-periodic branches. This ModernTSF entry exposes one adapted ASTGCN branch through the covariate forecasting contract; it does not implement the paper's three-branch fusion."
paper:
  title: "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting"
  venue: "AAAI 2019"
  year: 2019
  url: "https://doi.org/10.1609/aaai.v33i01.3301922"
codebase:
  url: "https://github.com/guoshnBJTU/ASTGCN-r-pytorch"
  revision: "2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296"
  license: ""
  usage: reference-only
---
# ASTGCN

The ASTGCN paper proposes a graph traffic forecaster with spatial-temporal attention, Chebyshev graph convolution, temporal convolution, and a learned fusion of recent, daily-periodic, and weekly-periodic branches. This ModernTSF entry exposes one adapted ASTGCN branch through the covariate forecasting contract; it does not implement the paper's three-branch fusion.

<!-- model-card:canonical:start -->
## Method overview

The ASTGCN paper proposes a graph traffic forecaster with spatial-temporal attention, Chebyshev graph convolution, temporal convolution, and a learned fusion of recent, daily-periodic, and weekly-periodic branches.

## Core architecture

This ModernTSF entry exposes one adapted ASTGCN branch through the covariate forecasting contract; it does not implement the paper's three-branch fusion.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://doi.org/10.1609/aaai.v33i01.3301922); title: Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting; venue/year: AAAI 2019 / 2019
- [codebase](https://github.com/guoshnBJTU/ASTGCN-r-pytorch); revision: `2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296`; license: `not available`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ASTGCN.toml`](../../../configs/models/ASTGCN.toml).

## Differences

- Official reference: https://github.com/guoshnBJTU/ASTGCN-r-pytorch at `2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296` (no license file declared at that revision).
Implementation: **rewrite** (clean-room audit pending). The implementation is adapted from CauAir's baseline rather than directly vendored from the official repository, and no numerical parity result is recorded.
- Known differences: this entry runs one ASTGCN branch rather than the paper's fused recent, daily-periodic, and weekly-periodic branches; missing graph input falls back to a dense graph; paper-specific preprocessing and the masked training objective are not reproduced here.

## Shared components

- [`graph_utils`](../../components/graph_utils.py)
- [`marks`](../../components/marks.py)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `nb_block=2`, `K=3`, `nb_chev_filter=64`, `nb_time_filter=64`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting
- **Venue**: AAAI 2019
- **Published**: 2019
- **arXiv**: N/A

## Abstract
Forecasting the traffic flows is a critical issue for researchers and practitioners in the field of transportation. However, it is very challenging since the traffic flows usually show high nonlinearities and complex patterns. Most existing traffic flow prediction methods, lacking abilities of modeling the dynamic spatial-temporal correlations of traffic data, thus cannot yield satisfactory prediction results. In this paper, we propose a novel attention based spatial-temporal graph convolutional network (ASTGCN) model to solve traffic flow forecasting problem. ASTGCN mainly consists of three independent components to respectively model three temporal properties of traffic flows, i.e., recent, daily-periodic and weekly-periodic dependencies. More specifically, each component contains two major parts: 1) the spatial-temporal attention mechanism to effectively capture the dynamic spatialtemporal correlations in traffic data; 2) the spatial-temporal convolution which simultaneously employs graph convolutions to capture the spatial patterns and common standard convolutions to describe the temporal features. The output of the three components are weighted fused to generate the final prediction results. Experiments on two real-world datasets from the Caltrans Performance Measurement System (PeMS) demonstrate that the proposed ASTGCN model outperforms the state-of-the-art baselines.

## In ModernTSF
Default config: `configs/models/ASTGCN.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

- Official reference: https://github.com/guoshnBJTU/ASTGCN-r-pytorch at `2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296` (no license file declared at that revision).
Implementation: **rewrite** (clean-room audit pending). The implementation is adapted from CauAir's baseline rather than directly vendored from the official repository, and no numerical parity result is recorded.
- Known differences: this entry runs one ASTGCN branch rather than the paper's fused recent, daily-periodic, and weekly-periodic branches; missing graph input falls back to a dense graph; paper-specific preprocessing and the masked training objective are not reproduced here.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/GuoLFSW19,
  author       = {Shengnan Guo and
                  Youfang Lin and
                  Ning Feng and
                  Chao Song and
                  Huaiyu Wan},
  title        = {Attention Based Spatial-Temporal Graph Convolutional Networks for
                  Traffic Flow Forecasting},
  booktitle    = {The Thirty-Third {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2019, The Thirty-First Innovative Applications of Artificial Intelligence
                  Conference, {IAAI} 2019, The Ninth {AAAI} Symposium on Educational
                  Advances in Artificial Intelligence, {EAAI} 2019, Honolulu, Hawaii,
                  USA, January 27 - February 1, 2019},
  pages        = {922--929},
  publisher    = {{AAAI} Press},
  year         = {2019},
  url          = {https://doi.org/10.1609/aaai.v33i01.3301922},
  doi          = {10.1609/AAAI.V33I01.3301922},
  timestamp    = {Mon, 04 Sep 2023 12:29:24 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/GuoLFSW19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
