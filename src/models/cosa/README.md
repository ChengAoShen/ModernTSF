---
name: "COSA"
implementation: rewrite
summary: "COSA (Context-aware Output-Space Adapter) is a time series forecasting model that addresses performance degradation of deployed forecasters under non-stationarity and distribution shifts. It is a minimal, plug-and-play adapter that directly corrects predictions of a frozen base model through residual correction modulated by gating, using a lightweight context vector that summarizes statistics from recently observed ground truth. Only adapter parameters are updated at test time under a leakage-free protocol with an adaptive learning rate schedule."
paper:
  title: "COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting"
  venue: "ICLR 2026"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# COSA

COSA (Context-aware Output-Space Adapter) is a time series forecasting model that addresses performance degradation of deployed forecasters under non-stationarity and distribution shifts. It is a minimal, plug-and-play adapter that directly corrects predictions of a frozen base model through residual correction modulated by gating, using a lightweight context vector that summarizes statistics from recently observed ground truth. Only adapter parameters are updated at test time under a leakage-free protocol with an adaptive learning rate schedule.

<!-- model-card:canonical:start -->
## Method overview

COSA (Context-aware Output-Space Adapter) is a time series forecasting model that addresses performance degradation of deployed forecasters under non-stationarity and distribution shifts.

## Core architecture

It is a minimal, plug-and-play adapter that directly corrects predictions of a frozen base model through residual correction modulated by gating, using a lightweight context vector that summarizes statistics from recently observed ground truth. Only adapter parameters are updated at test time under a leakage-free protocol with an adaptive learning rate schedule.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting; venue/year: ICLR 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/COSA.toml`](../../../configs/models/COSA.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
Deployed time-series forecasters suffer performance degradation under non-stationarity and distribution shifts. Test-time adaptation (TTA) for time-series forecasting differs from vision TTA because ground truth becomes observable shortly after prediction. Existing time-series TTA methods typically employ dual input/output adapters that indirectly modify data distributions, making their effect on the frozen model difficult to analyze. We introduce the Context-aware Output-Space Adapter (COSA), a minimal, plug-and-play adapter that directly corrects predictions of a frozen base model. COSA performs residual correction modulated by gating, utilizing the original prediction and a lightweight context vector that summarizes statistics from recently observed ground truth. At test time, only the adapter parameters (linear layer and gating) are updated under a leakage-free protocol, using observed ground truth with an adaptive learning rate schedule for faster adaptation. Across diverse scenarios, COSA demonstrates substantial performance gains versus baselines without TTA (13.91∼17.03%) and SOTA TTA methods (10.48∼13.05%), with particularly large improvements at long horizons, while adding a reasonable level of parameters and negligible computational overhead. The simplicity of COSA makes it architecture-agnostic and deployment-friendly.

## In ModernTSF
Default config: `configs/models/COSA.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

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
