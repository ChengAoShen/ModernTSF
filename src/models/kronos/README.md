---
model: "Kronos"
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

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ShiFCZXZL26,
  author       = {Yu Shi and
                  Zongliang Fu and
                  Shuo Chen and
                  Bohan Zhao and
                  Wei Xu and
                  Changshui Zhang and
                  Jian Li},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Kronos: {A} Foundation Model for the Language of Financial Markets},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {25366--25373},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i30.39730},
  doi          = {10.1609/AAAI.V40I30.39730},
  timestamp    = {Wed, 03 Jun 2026 10:10:49 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/ShiFCZXZL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
