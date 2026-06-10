---
model: "TimeMixer"
forecasting_setting: "time_series"
config: "configs/models/TimeMixer.toml"
registry: "models.timemixer.registry"
paper_title: "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
venue: "ICLR 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2405.14616"
---
# TimeMixer

TimeMixer is a fully MLP-based model for both long-term and short-term time series forecasting. It decomposes and mixes temporal patterns across multiple sampling scales: a Past-Decomposable-Mixing (PDM) block separates and aggregates seasonal and trend components in fine-to-coarse and coarse-to-fine directions, while a Future-Multipredictor-Mixing (FMM) block ensembles scale-specific predictors to leverage complementary forecasting information.

## Paper
- **Title**: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.14616

## Abstract
Time series forecasting is widely used in extensive applications, such as traffic planning and weather forecasting. However, real-world time series usually present intricate temporal variations, making forecasting extremely challenging. Going beyond the mainstream paradigms of plain decomposition and multiperiodicity analysis, we analyze temporal variations in a novel view of multiscale-mixing, which is based on an intuitive but important observation that time series present distinct patterns in different sampling scales. The microscopic and the macroscopic information are reflected in fine and coarse scales respectively, and thereby complex variations can be inherently disentangled. Based on this observation, we propose TimeMixer as a fully MLP-based architecture with Past-Decomposable-Mixing (PDM) and Future-Multipredictor-Mixing (FMM) blocks to take full advantage of disentangled multiscale series in both past extraction and future prediction phases. Concretely, PDM applies the decomposition to multiscale series and further mixes the decomposed seasonal and trend components in fine-to-coarse and coarse-to-fine directions separately, which successively aggregates the microscopic seasonal and macroscopic trend information. FMM further ensembles multiple predictors to utilize complementary forecasting capabilities in multiscale observations. Consequently, TimeMixer is able to achieve consistent state-of-the-art performances in both long-term and short-term forecasting tasks with favorable run-time efficiency.

## In ModernTSF
Default config: `configs/models/TimeMixer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/WangWSHLMZ024,
  author       = {Shiyu Wang and
                  Haixu Wu and
                  Xiaoming Shi and
                  Tengge Hu and
                  Huakun Luo and
                  Lintao Ma and
                  James Y. Zhang and
                  Jun Zhou},
  title        = {TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=7oLshfEIC2},
  timestamp    = {Thu, 22 May 2025 17:08:34 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/WangWSHLMZ024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
