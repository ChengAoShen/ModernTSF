---
model: "SOFTS"
forecasting_setting: "time_series"
config: "configs/models/SOFTS.toml"
registry: "models.softs.registry"
paper_title: "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion"
venue: "NeurIPS 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2404.14197"
---
# SOFTS

SOFTS (Series-cOre Fused Time Series forecaster) is an MLP-based model for multivariate time-series forecasting in the standard time-series setting. Its key innovation is the STar Aggregate-Redistribute (STAR) module, which uses a centralized strategy to model inter-channel dependencies: all series are aggregated into a single global core representation, which is then fused back with each individual series, achieving linear-complexity channel interaction without relying on distributed attention mechanisms.

## Paper
- **Title**: SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-04)
- **arXiv**: https://arxiv.org/abs/2404.14197

## Abstract
Multivariate time series forecasting plays a crucial role in various fields such as finance, traffic management, energy, and healthcare. Recent studies have highlighted the advantages of channel independence to resist distribution drift but neglect channel correlations, limiting further enhancements. Several methods utilize mechanisms like attention or mixer to address this by capturing channel correlations, but they either introduce excessive complexity or rely too heavily on the correlation to achieve satisfactory results under distribution drifts, particularly with a large number of channels. Addressing this gap, this paper presents an efficient MLP-based model, the Series-cOre Fused Time Series forecaster (SOFTS), which incorporates a novel STar Aggregate-Redistribute (STAR) module. Unlike traditional approaches that manage channel interactions through distributed structures, e.g., attention, STAR employs a centralized strategy to improve efficiency and reduce reliance on the quality of each channel. It aggregates all series to form a global core representation, which is then dispatched and fused with individual series representations to facilitate channel interactions effectively. SOFTS achieves superior performance over existing state-of-the-art methods with only linear complexity. The broad applicability of the STAR module across different forecasting models is also demonstrated empirically. For further research and development, we have made our code publicly available.

## In ModernTSF
Default config: `configs/models/SOFTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/LuCYZ24,
  author       = {Lu Han and
                  Xu{-}Yang Chen and
                  Han{-}Jia Ye and
                  De{-}Chuan Zhan},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {{SOFTS:} Efficient Multivariate Time Series Forecasting with Series-Core
                  Fusion},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/LuCYZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
