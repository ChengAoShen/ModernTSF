---
model: "MGSFformer"
forecasting_setting: "spatiotemporal"
config: "configs/models/MGSFformer.toml"
spec: "models.mgsfformer.spec"
paper_title: "MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction"
venue: "Information Fusion 2025"
year: 2025
arxiv: "https://doi.org/10.1016/j.inffus.2024.102607"
---
# MGSFformer

MGSFformer is a Multi-Granularity Spatiotemporal Fusion Transformer designed for node-level air quality prediction. It consists of three specialised sub-modules: a residual de-redundant block that eliminates information redundancy between data of different temporal granularities, a spatiotemporal attention block that captures correlations across monitoring stations and time, and a dynamic fusion block that adaptively weights and integrates multi-granularity predictions.

## Paper
- **Title**: MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction
- **Venue**: Information Fusion 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Air quality prediction is a critical task in environmental science. Air monitoring stations typically collect data at multiple sampling intervals (multiple granularities), each exhibiting distinct temporal patterns, and data from different stations exhibit strong spatiotemporal correlations. MGSFformer addresses both challenges simultaneously through three components: (1) a residual de-redundant block that removes redundant information across granularities, preventing the model from being misled by overlapping signals; (2) a spatiotemporal attention block that models correlations among stations and across time steps; and (3) a dynamic fusion block that assesses the relative importance of each granularity and integrates the resulting predictions. Experiments on three real-world air quality datasets demonstrate that MGSFformer outperforms 11 state-of-the-art baselines by approximately 5%.

## In ModernTSF
Default config: `configs/models/MGSFformer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

The bundled core maps the author repository's architecture, IE, STA, DF, and
RevIN modules at revision `ff665a422a0ae001cfdd1b60ec9b4338a5ab406e` into a
single module. It uses historical target values only—not future covariates—and
requires `seq_len` to be a multiple of 24. The author repository declares no
code license and no checkpoint parity has been established, so the entry
remains **unverified**.

## Citation

```bibtex
@article{DBLP:journals/inffus/YuWWSSYX25,
  author       = {Chengqing Yu and
                  Fei Wang and
                  Yilun Wang and
                  Zezhi Shao and
                  Tao Sun and
                  Di Yao and
                  Yongjun Xu},
  title        = {MGSFformer: {A} Multi-Granularity Spatiotemporal Fusion Transformer
                  for air quality prediction},
  journal      = {Inf. Fusion},
  volume       = {113},
  pages        = {102607},
  year         = {2025},
  url          = {https://doi.org/10.1016/j.inffus.2024.102607},
  doi          = {10.1016/J.INFFUS.2024.102607},
  timestamp    = {Sat, 31 May 2025 23:16:07 +0200},
  biburl       = {https://dblp.org/rec/journals/inffus/YuWWSSYX25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
