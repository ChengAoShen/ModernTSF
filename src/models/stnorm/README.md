---
name: "STNorm"
implementation: upstream
summary: "STNorm is a spatiotemporal forecasting model that augments a WaveNet-style backbone with two dedicated normalization modules — spatial normalization and temporal normalization — to separately refine high-frequency temporal components and local spatial components in multi-variate time-series data. It operates on node-structured data and does not require an externally provided static adjacency matrix."
paper:
  title: "ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting"
  venue: "KDD 2021"
  year: 2021
  url: "https://doi.org/10.1145/3447548.3467330"
codebase:
  url: "https://github.com/GestaltCogTeam/BasicTS"
  revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
  license: "Apache-2.0"
  usage: ported
---
# STNorm

STNorm is a spatiotemporal forecasting model that augments a WaveNet-style backbone with two dedicated normalization modules — spatial normalization and temporal normalization — to separately refine high-frequency temporal components and local spatial components in multi-variate time-series data. It operates on node-structured data and does not require an externally provided static adjacency matrix.

<!-- model-card:canonical:start -->
## Method overview

STNorm is a spatiotemporal forecasting model that augments a WaveNet-style backbone with two dedicated normalization modules — spatial normalization and temporal normalization — to separately refine high-frequency temporal components and local spatial components in multi-variate time-series data.

## Core architecture

It operates on node-structured data and does not require an externally provided static adjacency matrix.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Graph adjacency is supplied at construction; temporal/node covariates follow the runtime batch contract.

## Paper and code

- [paper](https://doi.org/10.1145/3447548.3467330); title: ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting; venue/year: KDD 2021 / 2021
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`; usage: `ported`

## Local implementation

This card declares a `upstream` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/STNorm.toml`](../../../configs/models/STNorm.toml).

## Differences

Implementation: **upstream** (numerical parity passed), pinned to `GestaltCogTeam/BasicTS@c218c07b6ce5e4cf908b147fd180c486346fed9c` (Apache-2.0). Exact-source eval/train outputs, spatial/temporal normalization intermediates, input gradients, every active parameter gradient, preprocessing, running buffers, and serialization match. Spatial and temporal normalization on the WaveNet backbone are retained; the preset and runner differ from the official experiments. The terminal residual projection is omitted only for the last layer: parity evidence confirms it has no gradient and cannot affect the skip-path prediction head.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `channels=16`, `kernel_size=2`, `blocks=2`, `layers=2`, `tnorm_bool=True`, `snorm_bool=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting
- **Venue**: KDD 2021
- **Published**: 2021
- **arXiv**: N/A

## Abstract
Multi-variate time series (MTS) data is generated from hybrid dynamical systems with unknown dynamics. The hybrid nature of such systems is a result of complex external impacts, which can be summarized as high-frequency and low-frequency from the temporal view, or global and local if we take the spatial view. These impacts are paramount to capture in time series forecasting tasks. In this paper, we propose temporal and spatial normalization modules which separately refine the high-frequency component and the local component underlying the raw data and can be integrated into canonical deep learning architectures such as WaveNet and Transformer. We conduct extensive experiments to demonstrate that the proposed method achieves superior performance on two public traffic network datasets, METR-LA and PEMS-BAY.

## In ModernTSF
Default config: `configs/models/STNorm.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

Implementation: **upstream** (numerical parity passed), pinned to `GestaltCogTeam/BasicTS@c218c07b6ce5e4cf908b147fd180c486346fed9c` (Apache-2.0). Exact-source eval/train outputs, spatial/temporal normalization intermediates, input gradients, every active parameter gradient, preprocessing, running buffers, and serialization match. Spatial and temporal normalization on the WaveNet backbone are retained; the preset and runner differ from the official experiments. The terminal residual projection is omitted only for the last layer: parity evidence confirms it has no gradient and cannot affect the skip-path prediction head.

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
