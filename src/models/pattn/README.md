---
model: "PAttn"
forecasting_setting: "time_series"
config: "configs/models/PAttn.toml"
spec: "models.pattn.spec"
paper_title: "Are Language Models Actually Useful for Time Series Forecasting?"
venue: "NeurIPS 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2406.16964"
---
# PAttn

PAttn is a deliberately simple patch-based Transformer baseline for time-series forecasting, introduced in the NeurIPS 2024 Spotlight paper "Are Language Models Actually Useful for Time Series Forecasting?". It pads and unfolds the input into overlapping patches, linearly embeds each patch per channel, processes the patch tokens with a single self-attention encoder block, then flattens and linearly projects to the forecast horizon, demonstrating that this minimal architecture matches or exceeds much heavier LLM-based forecasters.

## Paper
- **Title**: Are Language Models Actually Useful for Time Series Forecasting?
- **Venue**: NeurIPS 2024 (Spotlight)
- **Published**: 2024 (arXiv: 2024-06)
- **arXiv**: https://arxiv.org/abs/2406.16964

## Abstract
Large language models (LLMs) are being applied to time series forecasting. But are language models actually useful for time series? In a series of ablation studies on three recent and popular LLM-based time series forecasting methods, we find that removing the LLM component or replacing it with a basic attention layer does not degrade forecasting performance -- in most cases, the results even improve! We also find that despite their significant computational cost, pretrained LLMs do no better than models trained from scratch, do not represent the sequential dependencies in time series, and do not assist in few-shot settings. Additionally, we explore time series encoders and find that patching and attention structures perform similarly to LLM-based forecasters.

## In ModernTSF
Default config: `configs/models/PAttn.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

The implementation is adapted from MIT-licensed THUML Time-Series-Library commit `4e938a1767106324dd753b2a44832bf870a0252e` and cross-checked against the paper repository at `23bb8d5aa0b214056c4472e325c2d7977c1572ef`. Inert channel-count and attention-factor options were removed. Status: `adaptation`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/TanMGAH24,
  author       = {Mingtian Tan and
                  Mike A. Merrill and
                  Vinayak Gupta and
                  Tim Althoff and
                  Tom Hartvigsen},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {Are Language Models Actually Useful for Time Series Forecasting?},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/TanMGAH24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
