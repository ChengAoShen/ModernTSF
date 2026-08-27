---
name: "GOTSF"
implementation: rewrite
summary: "GOTSF is a time series forecasting model built around a goal-oriented training methodology that enables forecasting models to bias their predictions toward application-specified regions of interest at inference time without retraining. The model partitions the prediction space into fine-grained segments during training, which are dynamically reweighted and aggregated at inference time to emphasize whichever target range an application specifies."
paper:
  title: "Goal-Oriented Time-Series Forecasting: Foundation Framework Design"
  venue: "AAAI 2026"
  year: 2026
  url: "https://arxiv.org/abs/2504.17493"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# GOTSF

GOTSF is a time series forecasting model built around a goal-oriented training methodology that enables forecasting models to bias their predictions toward application-specified regions of interest at inference time without retraining. The model partitions the prediction space into fine-grained segments during training, which are dynamically reweighted and aggregated at inference time to emphasize whichever target range an application specifies.

<!-- model-card:canonical:start -->
## Method overview

GOTSF is a time series forecasting model built around a goal-oriented training methodology that enables forecasting models to bias their predictions toward application-specified regions of interest at inference time without retraining.

## Core architecture

The model partitions the prediction space into fine-grained segments during training, which are dynamically reweighted and aggregated at inference time to emphasize whichever target range an application specifies.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2504.17493); title: Goal-Oriented Time-Series Forecasting: Foundation Framework Design; venue/year: AAAI 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/GOTSF.toml`](../../../configs/models/GOTSF.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Goal-Oriented Time-Series Forecasting: Foundation Framework Design
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-04)
- **arXiv**: https://arxiv.org/abs/2504.17493

## Abstract
Conventional time-series forecasting methods typically aim to minimize overall prediction error, without accounting for the varying importance of different forecast ranges in downstream applications. We propose a training methodology that enables forecasting models to adapt their focus to application-specific regions of interest at inference time, without retraining. The approach partitions the prediction space into fine-grained segments during training, which are dynamically reweighted and aggregated to emphasize the target range specified by the application. Unlike prior methods that predefine these ranges, our framework supports flexible, on-demand adjustments. Experiments on standard benchmarks and a newly collected wireless communication dataset demonstrate that our method not only improves forecast accuracy within regions of interest but also yields measurable gains in downstream task performance. These results highlight the potential for closer integration between predictive modeling and decision-making in real-world systems.

## In ModernTSF
Default config: `configs/models/GOTSF.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

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
