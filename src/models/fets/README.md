---
model: "FeTS"
forecasting_setting: "time_series"
config: "configs/models/FeTS.toml"
registry: "models.fets.registry"
paper_title: "FeTS: A Feature-Aware Framework for Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
arxiv: ""
---
# FeTS

FeTS (Feature-Aware Framework for Time Series) is a multivariate time-series forecasting model accepted at AAAI 2026. It learns adaptive temporal importance weightings over input feature-time combinations to selectively emphasize the most informative dimensions, improving forecasting accuracy across standard benchmarks in the standard time-series forecasting setting.

## Paper
- **Title**: FeTS: A Feature-Aware Framework for Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
FeTS is a feature-aware forecasting framework for multivariate time series that learns adaptive importance weightings over input feature-time combinations. By selectively amplifying the most informative feature dimensions while suppressing irrelevant ones, FeTS improves forecasting accuracy across standard benchmarks. The framework is trained end-to-end and integrates with common backbone architectures, enabling efficient parameter utilization and competitive performance in long-term forecasting settings. (No arXiv preprint was found; this description is based on the AAAI 2026 acceptance and the official implementation repository at https://github.com/lllucky111/FeTS.)

## In ModernTSF
Default config: `configs/models/FeTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/WangCL26,
  author       = {Le Wang and
                  Jianyong Chen and
                  Songbai Liu},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {FeTS: {A} Feature-Aware Framework for Time Series Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {26328--26336},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i31.39838},
  doi          = {10.1609/AAAI.V40I31.39838},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/WangCL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
