---
model: "GOTSF"
forecasting_setting: "time_series"
config: "configs/models/GOTSF.toml"
registry: "models.gotsf.registry"
paper_title: "Goal-Oriented Time-Series Forecasting: Foundation Framework Design"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2504.17493"
---
# GOTSF

GOTSF is a time series forecasting model built around a goal-oriented training methodology that enables forecasting models to bias their predictions toward application-specified regions of interest at inference time without retraining. The model partitions the prediction space into fine-grained segments during training, which are dynamically reweighted and aggregated at inference time to emphasize whichever target range an application specifies.

## Paper
- **Title**: Goal-Oriented Time-Series Forecasting: Foundation Framework Design
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-04)
- **arXiv**: https://arxiv.org/abs/2504.17493

## Abstract
Conventional time-series forecasting methods typically aim to minimize overall prediction error, without accounting for the varying importance of different forecast ranges in downstream applications. We propose a training methodology that enables forecasting models to adapt their focus to application-specific regions of interest at inference time, without retraining. The approach partitions the prediction space into fine-grained segments during training, which are dynamically reweighted and aggregated to emphasize the target range specified by the application. Unlike prior methods that predefine these ranges, our framework supports flexible, on-demand adjustments. Experiments on standard benchmarks and a newly collected wireless communication dataset demonstrate that our method not only improves forecast accuracy within regions of interest but also yields measurable gains in downstream task performance. These results highlight the potential for closer integration between predictive modeling and decision-making in real-world systems.

## In ModernTSF
Default config: `configs/models/GOTSF.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/FecheteSAPLDS26,
  author       = {Luca{-}Andrei Fechete and
                  Mohamed Sana and
                  Fadhel Ayed and
                  Nicola Piovesan and
                  Wenjie Li and
                  Antonio De Domenico and
                  Tareq Si Salem},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Goal-Oriented Time-Series Forecasting: Foundation Framework Design},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {21065--21073},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i25.39249},
  doi          = {10.1609/AAAI.V40I25.39249},
  timestamp    = {Fri, 27 Mar 2026 17:13:39 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/FecheteSAPLDS26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
