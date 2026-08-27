---
name: "ImplicitForecaster"
implementation: rewrite
summary: "ImplicitForecaster (IF) is a time-series forecasting decoding module accepted at NeurIPS 2025. Rather than generating long-horizon forecasts by independently predicting each time point, it implicitly decomposes the target sequence into constituent waves parameterized by frequency, amplitude, and phase, capturing both long-term and short-term dynamics in a holistic manner and consistently boosting mainstream backbone models."
paper:
  title: "Towards Accurate Time Series Forecasting via Implicit Decoding"
  venue: "NeurIPS 2025"
  year: 2025
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# ImplicitForecaster

ImplicitForecaster (IF) is a time-series forecasting decoding module accepted at NeurIPS 2025. Rather than generating long-horizon forecasts by independently predicting each time point, it implicitly decomposes the target sequence into constituent waves parameterized by frequency, amplitude, and phase, capturing both long-term and short-term dynamics in a holistic manner and consistently boosting mainstream backbone models.

<!-- model-card:canonical:start -->
## Method overview

ImplicitForecaster (IF) is a time-series forecasting decoding module accepted at NeurIPS 2025.

## Core architecture

Rather than generating long-horizon forecasts by independently predicting each time point, it implicitly decomposes the target sequence into constituent waves parameterized by frequency, amplitude, and phase, capturing both long-term and short-term dynamics in a holistic manner and consistently boosting mainstream backbone models.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Towards Accurate Time Series Forecasting via Implicit Decoding; venue/year: NeurIPS 2025 / 2025
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ImplicitForecaster.toml`](../../../configs/models/ImplicitForecaster.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Towards Accurate Time Series Forecasting via Implicit Decoding
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Recent booming time series models have demonstrated remarkable forecasting performance. However, these methods often place greater focus on more effectively modelling the historical series, largely neglecting the forecasting phase, which generates long-term forecasts by separately predicting multiple time points. Given that real-world time series typically consist of various long short-term dynamics, independent predictions over individual time points may fail to express complex underlying patterns and can lead to a lack of global views. To address these issues, this work explores new perspectives from the forecasting phase and proposes a novel Implicit Forecaster (IF) as an additional decoding module. Inspired by decomposition forecasting, IF adopts a more nuanced approach by implicitly predicting constituent waves represented by their frequency, amplitude, and phase, thereby accurately forming the time series. Extensive experimental results from multiple real-world datasets show that IF can consistently boost mainstream time series models, achieving state-of-the-art forecasting performance.

## In ModernTSF
Default config: `configs/models/ImplicitForecaster.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

The official project does not currently publish a paper BibTeX entry or a
stable proceedings identifier. Until one is available, cite the official
software repository without inventing paper metadata:

```bibtex
@misc{implicitforecaster2025software,
  author       = {{Implicit Forecaster Contributors}},
  title        = {Towards Accurate Time Series Forecasting via Implicit Decoding},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/rakuyorain/Implicit-Forecaster}
}
```
