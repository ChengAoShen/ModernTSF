---
model: "SegRNN"
forecasting_setting: "time_series"
config: "configs/models/SegRNN.toml"
registry: "models.segrnn.registry"
paper_title: "SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting"
venue: "arXiv preprint"
year: 2023
arxiv: "https://arxiv.org/abs/2308.11200"
---
# SegRNN

SegRNN is an RNN-based model for long-term multivariate time-series forecasting that replaces the traditional point-wise recurrence with two complementary strategies: Segment-wise Iterations, which process fixed-length segments rather than individual time steps, and Parallel Multi-step Forecasting (PMF), which generates all future steps in a single parallel pass instead of autoregressively. Together these strategies drastically reduce the number of recurrent iterations, cutting runtime and memory by more than 78% compared to standard RNNs while outperforming Transformer-based competitors.

## Paper
- **Title**: SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2023 (arXiv: 2023-08)
- **arXiv**: https://arxiv.org/abs/2308.11200

## Abstract
RNN-based methods have faced challenges in the Long-term Time Series Forecasting (LTSF) domain when dealing with excessively long look-back windows and forecast horizons. Consequently, the dominance in this domain has shifted towards Transformer, MLP, and CNN approaches. The substantial number of recurrent iterations are the fundamental reasons behind the limitations of RNNs in LTSF. To address these issues, we propose two novel strategies to reduce the number of iterations in RNNs for LTSF tasks: Segment-wise Iterations and Parallel Multi-step Forecasting (PMF). RNNs that combine these strategies, namely SegRNN, significantly reduce the required recurrent iterations for LTSF, resulting in notable improvements in forecast accuracy and inference speed. Extensive experiments demonstrate that SegRNN not only outperforms SOTA Transformer-based models but also reduces runtime and memory usage by more than 78%. These achievements provide strong evidence that RNNs continue to excel in LTSF tasks and encourage further exploration of this domain with more RNN-based approaches.

## In ModernTSF
Default config: `configs/models/SegRNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/iotj/LinLWZMZ26,
  author       = {Shengsheng Lin and
                  Weiwei Lin and
                  Wentai Wu and
                  Feiyu Zhao and
                  Ruichao Mo and
                  Haotong Zhang},
  title        = {SegRNN: Segment Recurrent Neural Network for Long-Term Time-Series
                  Forecasting},
  journal      = {{IEEE} Internet Things J.},
  volume       = {13},
  number       = {5},
  pages        = {9861--9871},
  year         = {2026},
  url          = {https://doi.org/10.1109/JIOT.2025.3647705},
  doi          = {10.1109/JIOT.2025.3647705},
  timestamp    = {Wed, 11 Mar 2026 08:24:56 +0100},
  biburl       = {https://dblp.org/rec/journals/iotj/LinLWZMZ26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
