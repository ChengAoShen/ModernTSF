---
name: "STGCN"
implementation: upstream
summary: "STGCN (Spatio-Temporal Graph Convolutional Network) is a deep learning framework for node-level spatiotemporal forecasting, originally developed for traffic speed prediction. It combines graph convolution layers that capture spatial dependencies between nodes on a road network with temporal convolution layers that model short- and long-range time patterns, using fully convolutional structures to achieve fast training and compact parameterisation compared to recurrent alternatives."
paper:
  title: "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"
  venue: "IJCAI 2018"
  year: 2018
  url: "https://arxiv.org/abs/1709.04875"
codebase:
  url: "https://github.com/GestaltCogTeam/BasicTS"
  revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
  license: "Apache-2.0"
  usage: ported
---
# STGCN

STGCN (Spatio-Temporal Graph Convolutional Network) is a deep learning framework for node-level spatiotemporal forecasting, originally developed for traffic speed prediction. It combines graph convolution layers that capture spatial dependencies between nodes on a road network with temporal convolution layers that model short- and long-range time patterns, using fully convolutional structures to achieve fast training and compact parameterisation compared to recurrent alternatives.

<!-- model-card:canonical:start -->
## Method overview

STGCN (Spatio-Temporal Graph Convolutional Network) is a deep learning framework for node-level spatiotemporal forecasting, originally developed for traffic speed prediction.

## Core architecture

It combines graph convolution layers that capture spatial dependencies between nodes on a road network with temporal convolution layers that model short- and long-range time patterns, using fully convolutional structures to achieve fast training and compact parameterisation compared to recurrent alternatives.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Graph adjacency is supplied at construction; temporal/node covariates follow the runtime batch contract.

## Paper and code

- [paper](https://arxiv.org/abs/1709.04875); title: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting; venue/year: IJCAI 2018 / 2018
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`; usage: `ported`

## Local implementation

This card declares a `upstream` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/STGCN.toml`](../../../configs/models/STGCN.toml).

## Differences

Implementation: **upstream**; [numerical parity passed](../../../verification/parity/STGCN.json) against `GestaltCogTeam/BasicTS@c218c07b6ce5e4cf908b147fd180c486346fed9c` (Apache-2.0). Gated temporal and Chebyshev graph convolution blocks are retained with dataset adjacency converted to the BasicTS graph shift operator. Alignment convolutions are registered only on the channel-shrinking path that actually calls them.

## Shared components

- [`adj_norm`](../../components/adj_norm.py)
- [`marks`](../../components/marks.py)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `Kt=3`, `Ks=3`, `hidden_dim=32`, `bottleneck_dim=8`, `out_hidden_dim=32`, `act_func='glu'`, `graph_conv_type='cheb_graph_conv'`, `bias=True`, `droprate=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
- **Venue**: IJCAI 2018
- **Published**: 2018 (arXiv: 2017-09)
- **arXiv**: https://arxiv.org/abs/1709.04875

## Abstract
Timely accurate traffic forecast is crucial for urban traffic control and guidance. Due to the high nonlinearity and complexity of traffic flow, traditional methods cannot satisfy the requirements of mid-and-long term prediction tasks and often neglect spatial and temporal dependencies. In this paper, we propose a novel deep learning framework, Spatio-Temporal Graph Convolutional Networks (STGCN), to tackle the time series prediction problem in traffic domain. Instead of applying regular convolutional and recurrent units, we formulate the problem on graphs and build the model with complete convolutional structures, which enable much faster training speed with fewer parameters. Experiments show that our model STGCN effectively captures comprehensive spatio-temporal correlations through modeling multi-scale traffic networks and consistently outperforms state-of-the-art baselines on various real-world traffic datasets.

## In ModernTSF
Default config: `configs/models/STGCN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

Implementation: **upstream**; [numerical parity passed](../../../verification/parity/STGCN.json) against `GestaltCogTeam/BasicTS@c218c07b6ce5e4cf908b147fd180c486346fed9c` (Apache-2.0). Gated temporal and Chebyshev graph convolution blocks are retained with dataset adjacency converted to the BasicTS graph shift operator. Alignment convolutions are registered only on the channel-shrinking path that actually calls them.

## Citation

```bibtex
@inproceedings{DBLP:conf/ijcai/YuYZ18,
  author       = {Bing Yu and
                  Haoteng Yin and
                  Zhanxing Zhu},
  editor       = {J{\'{e}}r{\^{o}}me Lang},
  title        = {Spatio-Temporal Graph Convolutional Networks: {A} Deep Learning Framework
                  for Traffic Forecasting},
  booktitle    = {Proceedings of the Twenty-Seventh International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2018, July 13-19, 2018, Stockholm,
                  Sweden},
  pages        = {3634--3640},
  publisher    = {ijcai.org},
  year         = {2018},
  url          = {https://doi.org/10.24963/ijcai.2018/505},
  doi          = {10.24963/IJCAI.2018/505},
  timestamp    = {Sun, 04 Aug 2024 19:36:39 +0200},
  biburl       = {https://dblp.org/rec/conf/ijcai/YuYZ18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
