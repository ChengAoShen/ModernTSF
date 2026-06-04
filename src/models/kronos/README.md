---
model: "Kronos"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/Kronos.toml"
registry: "models.kronos.registry"
paper_title: "Kronos: A Foundation Model for the Language of Financial Markets"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2508.02739"
---
# Kronos

Kronos is a decoder-only foundation model pre-trained on over 12 billion financial candlestick (K-line) records from 45 global exchanges, covering tasks including price-series forecasting, volatility prediction, and synthetic market-data generation. In ModernTSF, a lightweight prompt-conditioned adapter captures the temporal inductive bias of the upstream model for general time-series forecasting using the standard RecentTSF training interface.

## Paper
- **Title**: Kronos: A Foundation Model for the Language of Financial Markets
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-08)
- **arXiv**: https://arxiv.org/abs/2508.02739

## Abstract
The success of large-scale pre-training paradigm, exemplified by Large Language Models (LLMs), has inspired the development of Time Series Foundation Models (TSFMs). However, their application to financial candlestick (K-line) data remains limited, often underperforming non-pre-trained architectures. Moreover, existing TSFMs often overlook crucial downstream tasks such as volatility prediction and synthetic data generation. To address these limitations, we propose Kronos, a unified, scalable pre-training framework tailored to financial K-line modeling. Kronos introduces a specialized tokenizer that discretizes continuous market information into token sequences, preserving both price dynamics and trade activity patterns. We pre-train Kronos using an autoregressive objective on a massive, multi-market corpus of over 12 billion K-line records from 45 global exchanges, enabling it to learn nuanced temporal and cross-asset representations. Kronos excels in a zero-shot setting across a diverse set of financial tasks. On benchmark datasets, Kronos boosts price series forecasting RankIC by 93% over the leading TSFM and 87% over the best non-pre-trained baseline. It also achieves a 9% lower MAE in volatility forecasting and a 22% improvement in generative fidelity for synthetic K-line sequences. These results establish Kronos as a robust, versatile foundation model for end-to-end financial time series analysis.

## In ModernTSF
Default config: `configs/models/Kronos.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
