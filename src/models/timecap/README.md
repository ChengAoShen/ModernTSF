---
model: "TimeCAP"
forecasting_setting: "time_series"
config: "configs/models/TimeCAP.toml"
registry: "models.timecap.registry"
paper_title: "TimeCAP: A Channel-Aware Pre-Training Framework for Multivariate Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
arxiv: ""
---
# TimeCAP

TimeCAP is a time series forecasting model for multivariate sequence prediction. It is the first purely channel-aware pre-training framework for multivariate time series, systematically integrating complementary autoregressive and one-shot generative paradigms via a flexible channel-grouping learning approach and an adaptive meta-routing mechanism that captures both intra-group local patterns and global inter-channel coherence.

## Paper
- **Title**: TimeCAP: A Channel-Aware Pre-Training Framework for Multivariate Time Series Forecasting
- **Venue**: AAAI 2026 (Oral)
- **Published**: 2026
- **arXiv**: N/A

## Abstract
TimeCAP introduces the first purely channel-aware pre-training framework for multivariate time series, internalizing latent causal relationships among variables inherent in multi-domain data and effectively transferring the acquired knowledge to downstream applications. Existing approaches exhibit two critical limitations: underestimating the significance of multivariate dependencies in learning generalizable representations, and failing to reconcile the complementary strengths of autoregressive and one-shot generative paradigms. TimeCAP addresses both by presenting a flexible channel-grouping learning approach, complemented by an adaptive meta-routing mechanism, enabling the model to simultaneously recognize intra-group local patterns while maintaining global coherence. Intra- and inter-group multivariate dependencies are captured through self- and cross-attention with a channel-aware mask, which strictly confines interactions among time-aligned, fine-grained multivariate tokens. In few-shot evaluation, TimeCAP achieves average MSE and MAE reductions of 11.8% and 6% over leading baselines, while also outperforming state-of-the-art models in full-shot and zero-shot settings by large margins.

## In ModernTSF
Default config: `configs/models/TimeCAP.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
