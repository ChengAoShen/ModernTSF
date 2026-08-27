---
name: "FeTS"
implementation: rewrite
summary: "FeTS (Feature-Aware Framework for Time Series) is a multivariate time-series forecasting model accepted at AAAI 2026. It learns adaptive temporal importance weightings over input feature-time combinations to selectively emphasize the most informative dimensions, improving forecasting accuracy across standard benchmarks in the standard time-series forecasting setting."
paper:
  title: "FeTS: A Feature-Aware Framework for Time Series Forecasting"
  venue: "AAAI 2026"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# FeTS

FeTS (Feature-Aware Framework for Time Series) is a multivariate time-series forecasting model accepted at AAAI 2026. It learns adaptive temporal importance weightings over input feature-time combinations to selectively emphasize the most informative dimensions, improving forecasting accuracy across standard benchmarks in the standard time-series forecasting setting.

<!-- model-card:canonical:start -->
## Method overview

FeTS (Feature-Aware Framework for Time Series) is a multivariate time-series forecasting model accepted at AAAI 2026.

## Core architecture

It learns adaptive temporal importance weightings over input feature-time combinations to selectively emphasize the most informative dimensions, improving forecasting accuracy across standard benchmarks in the standard time-series forecasting setting.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: FeTS: A Feature-Aware Framework for Time Series Forecasting; venue/year: AAAI 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/FeTS.toml`](../../../configs/models/FeTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: FeTS: A Feature-Aware Framework for Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
FeTS is a feature-aware forecasting framework for multivariate time series that learns adaptive importance weightings over input feature-time combinations. By selectively amplifying the most informative feature dimensions while suppressing irrelevant ones, FeTS improves forecasting accuracy across standard benchmarks. The framework is trained end-to-end and integrates with common backbone architectures, enabling efficient parameter utilization and competitive performance in long-term forecasting settings. (No arXiv preprint was found; this description is based on the AAAI 2026 acceptance and the official implementation repository at https://github.com/lllucky111/FeTS.)

## In ModernTSF
Default config: `configs/models/FeTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

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
