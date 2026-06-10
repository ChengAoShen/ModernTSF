---
model: "PULSE"
forecasting_setting: "time_series"
config: "configs/models/PULSE.toml"
registry: "models.pulse.registry"
paper_title: "Generative Phase Evolution for Non-Stationary Time Series Forecasting"
venue: "ICML 2026"
year: 2026
arxiv: ""
---
# PULSE

PULSE is a physics-informed generative framework for non-stationary time-series forecasting. Instead of passively fitting historical patterns, it separates deterministic phase structures from stochastic fluctuations, generates future phase trajectories, and simulates residual distribution shifts — an approach that shifts forecasting from historical fitting to generative phase evolution. In ModernTSF, a lightweight adapter (RecentTSFModel, style="phase") captures this inductive bias within the standard training pipeline.

## Paper
- **Title**: Generative Phase Evolution for Non-Stationary Time Series Forecasting
- **Venue**: ICML 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
PULSE introduces a physics-informed framework that reframes time-series forecasting as a generative phase-evolution problem rather than a historical-fitting task. The method decomposes each series into a deterministic phase structure and stochastic residual fluctuations. Future phase trajectories are generated autoregressively, while a separate module simulates distribution shifts in the residual component, enabling the model to handle non-stationary dynamics that cause distribution shifts between training and inference. Evaluated across 12 real-world datasets covering 24 evaluation metrics, PULSE achieved the best result on 18 of 24 metrics, demonstrating strong generalization to unseen non-stationary conditions.

## In ModernTSF
Default config: `configs/models/PULSE.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{liu2026pulse,
  author    = {Yangyou Liu and Zezhi Shao and Xinyu Chen and Hu Chen and Fei Wang and Yuankai Wu},
  title     = {{PULSE}: Generative Phase Evolution for Non-Stationary Time Series Forecasting},
  booktitle = {Forty-Third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://github.com/Gemost/PULSE}
}
```
