---
model: "OccamVTS"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/OccamVTS.toml"
registry: "models.occamvts.registry"
paper_title: "OccamVTS: Distilling Vision Models to 1% Parameters for Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2508.01727"
---
# OccamVTS

OccamVTS is a knowledge-distillation-based time series forecasting model for the standard time-series setting. It reveals that 99% of large vision model (LVM) parameters are unnecessary for time series tasks and proposes a pyramid-style feature alignment combined with correlation and feature distillation to transfer only the essential low-level textural patterns from pre-trained LVMs into a compact lightweight network — improving accuracy by eliminating overfitting to irrelevant visual features while preserving essential temporal patterns.

## Paper
- **Title**: OccamVTS: Distilling Vision Models to 1% Parameters for Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-08)
- **arXiv**: https://arxiv.org/abs/2508.01727

## Abstract
Time series forecasting is fundamental to diverse applications, with recent approaches leverage large vision models (LVMs) to capture temporal patterns through visual representations. We reveal that while vision models enhance forecasting performance, 99% of their parameters are unnecessary for time series tasks. Through cross-modal analysis, we find that time series align with low-level textural features but not high-level semantics, which can impair forecasting accuracy. We propose OccamVTS, a knowledge distillation framework that extracts only the essential 1% of predictive information from LVMs into lightweight networks. Using pre-trained LVMs as privileged teachers, OccamVTS employs pyramid-style feature alignment combined with correlation and feature distillation to transfer beneficial patterns while filtering out semantic noise. Counterintuitively, this aggressive parameter reduction improves accuracy by eliminating overfitting to irrelevant visual features while preserving essential temporal patterns. Extensive experiments across multiple benchmark datasets demonstrate that OccamVTS consistently achieves state-of-the-art performance with only 1% of the original parameters, particularly excelling in few-shot and zero-shot scenarios.

## In ModernTSF
Default config: `configs/models/OccamVTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
