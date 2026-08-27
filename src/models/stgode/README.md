---
name: "STGODE"
implementation: rewrite
summary: "STGODE is a spatiotemporal learning model for node-structured traffic and graph data that captures continuous spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE). By coupling a semantic adjacency matrix with a temporal dilated convolution structure, it overcomes the over-smoothing limitation of shallow GNNs and captures both structural and semantic long-range dependencies between nodes."
paper:
  title: "Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting"
  venue: "KDD 2021"
  year: 2021
  url: "https://doi.org/10.1145/3447548.3467430"
codebase:
  url: "https://github.com/GestaltCogTeam/BasicTS"
  revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
  license: "Apache-2.0"
  usage: reference-only
---
# STGODE

STGODE is a spatiotemporal learning model for node-structured traffic and graph data that captures continuous spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE). By coupling a semantic adjacency matrix with a temporal dilated convolution structure, it overcomes the over-smoothing limitation of shallow GNNs and captures both structural and semantic long-range dependencies between nodes.

## Paper
- **Title**: Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting
- **Venue**: KDD 2021
- **Published**: 2021 (arXiv: 2021-06)
- **arXiv**: https://arxiv.org/abs/2106.12931

## Abstract
Spatial-temporal forecasting has attracted tremendous attention in a wide range of applications, and traffic flow prediction is a canonical and typical example. The complex and long-range spatial-temporal correlations of traffic flow bring it to a most intractable challenge. Existing works typically utilize shallow graph convolution networks (GNNs) and temporal extracting modules to model spatial and temporal dependencies respectively. However, the representation ability of such models is limited due to: (1) shallow GNNs are incapable to capture long-range spatial correlations, (2) only spatial connections are considered and a mass of semantic connections are ignored, which are of great importance for a comprehensive understanding of traffic networks. To this end, we propose Spatial-Temporal Graph Ordinary Differential Equation Networks (STGODE). Specifically, we capture spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE), as a result, deeper networks can be constructed and spatial-temporal features are utilized synchronously. To understand the network more comprehensively, semantical adjacency matrix is considered in our model, and a well-design temporal dialated convolution structure is used to capture long term temporal dependencies. We evaluate our model on multiple real-world traffic datasets and superior performance is achieved over state-of-the-art baselines.

## In ModernTSF
Default config: `configs/models/STGODE.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

Implementation: **rewrite** (clean-room audit pending). The licensed source is pinned to
[`GestaltCogTeam/BasicTS`](https://github.com/GestaltCogTeam/BasicTS) revision
`c218c07b6ce5e4cf908b147fd180c486346fed9c` under Apache-2.0 and traces to the
authors' [`square-coder/STGODE`](https://github.com/square-coder/STGODE)
implementation. The dual graph-ODE backbone and dilated temporal convolutions
are retained. ModernTSF substitutes normalized dataset adjacency for the DTW
semantic graph, and replaces `torchdiffeq`'s one-step Euler call with the
algebraically identical explicit update. It also fixes the upstream
conditional-expression precedence bug that bypassed temporal convolutions when
input and output widths matched. These changes, especially the semantic-graph
substitution, prevent an `upstream implementation` equivalence claim.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/FangLSX21,
  author       = {Zheng Fang and
                  Qingqing Long and
                  Guojie Song and
                  Kunqing Xie},
  editor       = {Feida Zhu and
                  Beng Chin Ooi and
                  Chunyan Miao},
  title        = {Spatial-Temporal Graph {ODE} Networks for Traffic Flow Forecasting},
  booktitle    = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
  pages        = {364--373},
  publisher    = {{ACM}},
  year         = {2021},
  url          = {https://doi.org/10.1145/3447548.3467430},
  doi          = {10.1145/3447548.3467430},
  timestamp    = {Tue, 29 Nov 2022 09:04:02 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/FangLSX21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
