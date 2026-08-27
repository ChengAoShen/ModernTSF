---
name: "FreTS"
implementation: rewrite
summary: "FreTS is a multivariate time series forecasting model that applies redesigned multi-layer perceptrons directly in the frequency domain, operating on both the real and imaginary components of the frequency spectrum to capture global dependencies and exploit the energy compaction property of the Fourier transform."
paper:
  title: "Frequency-domain MLPs are More Effective Learners in Time Series Forecasting"
  venue: "NeurIPS 2023"
  year: 2023
  url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html"
codebase:
  url: "https://github.com/aikunyi/FreTS"
  revision: "6de28ab19f83955087e2690cdfbb29b065ab0b9c"
  license: "Apache-2.0"
  usage: reference-only
---
# FreTS

FreTS is a multivariate time series forecasting model that applies redesigned multi-layer perceptrons directly in the frequency domain, operating on both the real and imaginary components of the frequency spectrum to capture global dependencies and exploit the energy compaction property of the Fourier transform.

<!-- model-card:canonical:start -->
## Method overview

FreTS is a multivariate time series forecasting model that applies redesigned multi-layer perceptrons directly in the frequency domain, operating on both the real and imaginary components of the frequency spectrum to capture global dependencies and exploit the energy compaction property of the Fourier transform.

## Core architecture

FreTS is a multivariate time series forecasting model that applies redesigned multi-layer perceptrons directly in the frequency domain, operating on both the real and imaginary components of the frequency spectrum to capture global dependencies and exploit the energy compaction property of the Fourier transform.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html); title: Frequency-domain MLPs are More Effective Learners in Time Series Forecasting; venue/year: NeurIPS 2023 / 2023
- [codebase](https://github.com/aikunyi/FreTS); revision: `6de28ab19f83955087e2690cdfbb29b065ab0b9c`; license: `Apache-2.0`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/FreTS.toml`](../../../configs/models/FreTS.toml).

## Differences

Compared against the author repository at commit `6de28ab19f83955087e2690cdfbb29b065ab0b9c` (Apache-2.0). The complex frequency-domain channel and temporal MLP stages are retained in a forecast-only benchmark adapter.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `embed_size=128`, `hidden_size=256`, `channel_independence=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Frequency-domain MLPs are More Effective Learners in Time Series Forecasting
- **Venue**: NeurIPS 2023
- **Published**: 2023 (arXiv: 2023-11)
- **arXiv**: https://arxiv.org/abs/2311.06184

## Abstract
Time series forecasting has played the key role in different industrial, including finance, traffic, energy, and healthcare domains. While existing literatures have designed many sophisticated architectures based on RNNs, GNNs, or Transformers, another kind of approaches based on multi-layer perceptrons (MLPs) are proposed with simple structure, low complexity, and superior performance. However, most MLP-based forecasting methods suffer from the point-wise mappings and information bottleneck, which largely hinders the forecasting performance. To overcome this problem, we explore a novel direction of applying MLPs in the frequency domain for time series forecasting. We investigate the learned patterns of frequency-domain MLPs and discover their two inherent characteristic benefiting forecasting, (i) global view: frequency spectrum makes MLPs own a complete view for signals and learn global dependencies more easily, and (ii) energy compaction: frequency-domain MLPs concentrate on smaller key part of frequency components with compact signal energy. Then, we propose FreTS, a simple yet effective architecture built upon Frequency-domain MLPs for Time Series forecasting. FreTS mainly involves two stages, (i) Domain Conversion, that transforms time-domain signals into complex numbers of frequency domain; (ii) Frequency Learning, that performs our redesigned MLPs for the learning of real and imaginary part of frequency components. The above stages operated on both inter-series and intra-series scales further contribute to channel-wise and time-wise dependency learning. Extensive experiments on 13 real-world benchmarks (including 7 benchmarks for short-term forecasting and 6 benchmarks for long-term forecasting) demonstrate our consistent superiority over state-of-the-art methods.

## In ModernTSF
Default config: `configs/models/FreTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

Compared against the author repository at commit `6de28ab19f83955087e2690cdfbb29b065ab0b9c` (Apache-2.0). The complex frequency-domain channel and temporal MLP stages are retained in a forecast-only benchmark adapter.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/YiZFWWHALCN23,
  author       = {Kun Yi and
                  Qi Zhang and
                  Wei Fan and
                  Shoujin Wang and
                  Pengyang Wang and
                  Hui He and
                  Ning An and
                  Defu Lian and
                  Longbing Cao and
                  Zhendong Niu},
  editor       = {Alice Oh and
                  Tristan Naumann and
                  Amir Globerson and
                  Kate Saenko and
                  Moritz Hardt and
                  Sergey Levine},
  title        = {Frequency-domain MLPs are More Effective Learners in Time Series Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                  on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                  LA, USA, December 10 - 16, 2023},
  year         = {2023},
  url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html},
  timestamp    = {Thu, 29 Aug 2024 14:25:54 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/YiZFWWHALCN23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
