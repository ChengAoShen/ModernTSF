---
model: "COSA"
forecasting_setting: "time_series"
config: "configs/models/COSA.toml"
registry: "models.cosa.registry"
paper_title: "COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting"
venue: "ICLR 2026"
year: 2026
arxiv: ""
---
# COSA

COSA (Context-aware Output-Space Adapter) is a time series forecasting model that addresses performance degradation of deployed forecasters under non-stationarity and distribution shifts. It is a minimal, plug-and-play adapter that directly corrects predictions of a frozen base model through residual correction modulated by gating, using a lightweight context vector that summarizes statistics from recently observed ground truth. Only adapter parameters are updated at test time under a leakage-free protocol with an adaptive learning rate schedule.

## Paper
- **Title**: COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
Deployed time-series forecasters suffer performance degradation under non-stationarity and distribution shifts. Test-time adaptation (TTA) for time-series forecasting differs from vision TTA because ground truth becomes observable shortly after prediction. Existing time-series TTA methods typically employ dual input/output adapters that indirectly modify data distributions, making their effect on the frozen model difficult to analyze. We introduce the Context-aware Output-Space Adapter (COSA), a minimal, plug-and-play adapter that directly corrects predictions of a frozen base model. COSA performs residual correction modulated by gating, utilizing the original prediction and a lightweight context vector that summarizes statistics from recently observed ground truth. At test time, only the adapter parameters (linear layer and gating) are updated under a leakage-free protocol, using observed ground truth with an adaptive learning rate schedule for faster adaptation. Across diverse scenarios, COSA demonstrates substantial performance gains versus baselines without TTA (13.91∼17.03%) and SOTA TTA methods (10.48∼13.05%), with particularly large improvements at long horizons, while adding a reasonable level of parameters and negligible computational overhead. The simplicity of COSA makes it architecture-agnostic and deployment-friendly.

## In ModernTSF
Default config: `configs/models/COSA.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{im2026cosa,
  author    = {Jeonghwan Im and Hyuk-Yoon Kwon},
  title     = {{COSA}: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=L7Z5wBMPrW}
}
```
