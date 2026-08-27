---
name: "STTN"
implementation: rewrite
summary: "STTN (Spatial-Temporal Transformer Networks) is a spatiotemporal forecasting model designed for node-structured traffic and sensor-network data. It combines a spatial Transformer that dynamically models directed spatial dependencies with a self-attention mechanism — capturing real-time node-to-node relationships without a fixed adjacency matrix — with a temporal Transformer that captures long-range bidirectional temporal dependencies, yielding competitive accuracy especially for long-horizon traffic flow forecasting."
paper:
  title: "Spatial-Temporal Transformer Networks for Traffic Flow Forecasting"
  venue: "arXiv preprint"
  year: 2020
  url: "https://arxiv.org/abs/2001.02908"
codebase:
  url: "https://github.com/xumingxingsjtu/STTN"
  revision: "d24f8d331a6d81b819cfe0a9430793ae028d25ad"
  license: "NOASSERTION"
  usage: reference-only
---
# STTN

STTN (Spatial-Temporal Transformer Networks) is a spatiotemporal forecasting model designed for node-structured traffic and sensor-network data. It combines a spatial Transformer that dynamically models directed spatial dependencies with a self-attention mechanism — capturing real-time node-to-node relationships without a fixed adjacency matrix — with a temporal Transformer that captures long-range bidirectional temporal dependencies, yielding competitive accuracy especially for long-horizon traffic flow forecasting.

<!-- model-card:canonical:start -->
## Method overview

STTN (Spatial-Temporal Transformer Networks) is a spatiotemporal forecasting model designed for node-structured traffic and sensor-network data.

## Core architecture

It combines a spatial Transformer that dynamically models directed spatial dependencies with a self-attention mechanism — capturing real-time node-to-node relationships without a fixed adjacency matrix — with a temporal Transformer that captures long-range bidirectional temporal dependencies, yielding competitive accuracy especially for long-horizon traffic flow forecasting.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Graph adjacency is supplied at construction; temporal/node covariates follow the runtime batch contract.

## Paper and code

- [paper](https://arxiv.org/abs/2001.02908); title: Spatial-Temporal Transformer Networks for Traffic Flow Forecasting; venue/year: arXiv preprint / 2020
- [codebase](https://github.com/xumingxingsjtu/STTN); revision: `d24f8d331a6d81b819cfe0a9430793ae028d25ad`; license: `NOASSERTION`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/STTN.toml`](../../../configs/models/STTN.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`diffusion_conv`](../../components/diffusion_conv.py)
- [`graph_utils`](../../components/graph_utils.py)
- [`marks`](../../components/marks.py)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `d_model=64`, `mlp_expand=4`, `num_layers=3`, `dropout=0.1`, `adj_type='doubletransition'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatial-Temporal Transformer Networks for Traffic Flow Forecasting
- **Venue**: arXiv preprint
- **Published**: 2020 (arXiv: 2020-01)
- **arXiv**: https://arxiv.org/abs/2001.02908

## Abstract
Traffic forecasting has emerged as a core component of intelligent transportation systems. However, timely accurate traffic forecasting, especially long-term forecasting, still remains an open challenge due to the highly nonlinear and dynamic spatial-temporal dependencies of traffic flows. In this paper, we propose a novel paradigm of Spatial-Temporal Transformer Networks (STTNs) that leverages dynamical directed spatial dependencies and long-range temporal dependencies to improve the accuracy of long-term traffic forecasting. Specifically, we present a new variant of graph neural networks, named spatial transformer, by dynamically modeling directed spatial dependencies with self-attention mechanism to capture realtime traffic conditions as well as the directionality of traffic flows. Furthermore, different spatial dependency patterns can be jointly modeled with multi-heads attention mechanism to consider diverse relationships related to different factors (e.g. similarity, connectivity and covariance). On the other hand, the temporal transformer is utilized to model long-range bidirectional temporal dependencies across multiple time steps. Finally, they are composed as a block to jointly model the spatial-temporal dependencies for accurate traffic prediction. Compared to existing works, the proposed model enables fast and scalable training over a long range spatial-temporal dependencies. Experiment results demonstrate that the proposed model achieves competitive results compared with the state-of-the-arts, especially forecasting long-term traffic flows on real-world PeMS-Bay and PeMSD7(M) datasets.

## In ModernTSF
Default config: `configs/models/STTN.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

This entry is an **pending verification CauAir-derived PyTorch baseline**, not a direct
port of the official TensorFlow repository. The local spatial block combines
attention with fixed-adjacency second-order graph convolution, creates a dense
graph when none is supplied, appends shared calendar covariates, and uses a
different output head. Attention is fixed at four heads; `mlp_expand` controls
only feed-forward width. The official revision `d24f8d331a6d81b819cfe0a9430793ae028d25ad`
and CauAir revision `73dae00ca6ad14abb15174a0a0286d500e868b94` both lack an
explicit code license, and no numerical parity is available.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2001-02908,
  author       = {Mingxing Xu and
                  Wenrui Dai and
                  Chunmiao Liu and
                  Xing Gao and
                  Weiyao Lin and
                  Guo{-}Jun Qi and
                  Hongkai Xiong},
  title        = {Spatial-Temporal Transformer Networks for Traffic Flow Forecasting},
  journal      = {CoRR},
  volume       = {abs/2001.02908},
  year         = {2020},
  url          = {http://arxiv.org/abs/2001.02908},
  eprinttype   = {arXiv},
  eprint       = {2001.02908},
  timestamp    = {Tue, 14 Jan 2020 10:25:48 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2001-02908.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
