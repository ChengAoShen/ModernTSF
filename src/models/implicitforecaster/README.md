---
model: "ImplicitForecaster"
forecasting_setting: "time_series"
config: "configs/models/ImplicitForecaster.toml"
registry: "models.implicitforecaster.registry"
paper_title: "Towards Accurate Time Series Forecasting via Implicit Decoding"
venue: "NeurIPS 2025"
year: 2025
arxiv: ""
---
# ImplicitForecaster

ImplicitForecaster (IF) is a time-series forecasting decoding module accepted at NeurIPS 2025. Rather than generating long-horizon forecasts by independently predicting each time point, it implicitly decomposes the target sequence into constituent waves parameterized by frequency, amplitude, and phase, capturing both long-term and short-term dynamics in a holistic manner and consistently boosting mainstream backbone models.

## Paper
- **Title**: Towards Accurate Time Series Forecasting via Implicit Decoding
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Recent booming time series models have demonstrated remarkable forecasting performance. However, these methods often place greater focus on more effectively modelling the historical series, largely neglecting the forecasting phase, which generates long-term forecasts by separately predicting multiple time points. Given that real-world time series typically consist of various long short-term dynamics, independent predictions over individual time points may fail to express complex underlying patterns and can lead to a lack of global views. To address these issues, this work explores new perspectives from the forecasting phase and proposes a novel Implicit Forecaster (IF) as an additional decoding module. Inspired by decomposition forecasting, IF adopts a more nuanced approach by implicitly predicting constituent waves represented by their frequency, amplitude, and phase, thereby accurately forming the time series. Extensive experimental results from multiple real-world datasets show that IF can consistently boost mainstream time series models, achieving state-of-the-art forecasting performance.

## In ModernTSF
Default config: `configs/models/ImplicitForecaster.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
