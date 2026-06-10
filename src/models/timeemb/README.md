---
model: "TimeEmb"
forecasting_setting: "time_series"
config: "configs/models/TimeEmb.toml"
registry: "models.timeemb.registry"
paper_title: "TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting"
venue: "arXiv preprint"
year: 2025
arxiv: "https://arxiv.org/abs/2510.00461"
---
# TimeEmb

TimeEmb is a lightweight time-series forecasting model that disentangles static (time-invariant) and dynamic (time-varying) components of a series. A global timestamp-aware embedding bank captures recurring stable patterns, while a frequency-domain filtering mechanism handles short-term fluctuations — the two streams are combined to produce multi-step forecasts. The model can also serve as a plug-in module to enhance existing forecasters with minimal overhead.

## Paper
- **Title**: TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.00461

## Abstract
Temporal non-stationarity, the phenomenon that time series distributions change over time, poses fundamental challenges to reliable time series forecasting. Intuitively, the complex time series can be decomposed into two factors, i.e. time-invariant and time-varying components, which indicate static and dynamic patterns, respectively. Nonetheless, existing methods often conflate the time-varying and time-invariant components, and jointly learn the combined long-term patterns and short-term fluctuations, leading to suboptimal performance facing distribution shifts. To address this issue, we initiatively propose a lightweight static-dynamic decomposition framework, TimeEmb, for time series forecasting. TimeEmb innovatively separates time series into two complementary components: (1) time-invariant component, captured by a novel global embedding module that learns persistent representations across time series, and (2) time-varying component, processed by an efficient frequency-domain filtering mechanism inspired by full-spectrum analysis in signal processing. Experiments on real-world datasets demonstrate that TimeEmb outperforms state-of-the-art baselines and requires fewer computational resources. We conduct comprehensive quantitative and qualitative analyses to verify the efficacy of static-dynamic disentanglement. This lightweight framework can also improve existing time-series forecasting methods with simple integration.

## In ModernTSF
Default config: `configs/models/TimeEmb.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{xia2025timeemb,
  author        = {Mingyuan Xia and
                  Chunxu Zhang and
                  Zijian Zhang and
                  Hao Miao and
                  Qidong Liu and
                  Yuanshao Zhu and
                  Bo Yang},
  title         = {TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting},
  year          = {2025},
  eprint        = {2510.00461},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.00461}
}
```
