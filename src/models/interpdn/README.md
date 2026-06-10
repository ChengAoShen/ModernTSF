---
model: "InterPDN"
forecasting_setting: "time_series"
config: "configs/models/InterPDN.toml"
registry: "models.interpdn.registry"
paper_title: "Time Series Forecasting via Direct Per-Step Probability Distribution Modeling"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2511.23260"
---
# InterPDN

InterPDN (interleaved dual-branch Probability Distribution Network) is a time series forecasting model for standard multivariate or univariate sequences. Rather than predicting a scalar at each future step, it directly constructs a discrete probability distribution per step; the regression output is computed as the expectation over a predefined support set. A dual-branch architecture with interleaved support sets, coarse temporal-scale branches for long-term trend, and self-supervised consistency constraints between branches further improves robustness.

## Paper
- **Title**: Time Series Forecasting via Direct Per-Step Probability Distribution Modeling
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-11)
- **arXiv**: https://arxiv.org/abs/2511.23260

## Abstract
Deep neural network-based time series prediction models have recently demonstrated superior capabilities in capturing complex temporal dependencies. However, it is challenging for these models to account for uncertainty associated with their predictions, because they directly output scalar values at each time step. To address such a challenge, we propose a novel model named interleaved dual-branch Probability Distribution Network (interPDN), which directly constructs discrete probability distributions per step instead of a scalar. The regression output at each time step is derived by computing the expectation of the predictive distribution on a predefined support set. To mitigate prediction anomalies, a dual-branch architecture is introduced with interleaved support sets, augmented by coarse temporal-scale branches for long-term trend forecasting. Outputs from another branch are treated as auxiliary signals to impose self-supervised consistency constraints on the current branch's prediction. Extensive experiments on multiple real-world datasets demonstrate the superior performance of interPDN.

## In ModernTSF
Default config: `configs/models/InterPDN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/KongH26,
  author       = {Linghao Kong and
                  Xiaopeng Hong},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Time Series Forecasting via Direct Per-Step Probability Distribution
                  Modeling},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {22653--22661},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i27.39426},
  doi          = {10.1609/AAAI.V40I27.39426},
  timestamp    = {Thu, 26 Mar 2026 16:46:49 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/KongH26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
