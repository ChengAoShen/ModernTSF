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

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/RenLHZZLLL26,
  author       = {Chuanru Ren and
                  Yao Lu and
                  Tianjin Huang and
                  Haowen Zheng and
                  Hengde Zhu and
                  Yunyin Li and
                  Hengxiao Li and
                  Lu Liu},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {TimeCAP: {A} Channel-Aware Pre-Training Framework for Multivariate
                  Time Series Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {25108--25116},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i30.39700},
  doi          = {10.1609/AAAI.V40I30.39700},
  timestamp    = {Fri, 27 Mar 2026 07:38:55 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/RenLHZZLLL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
