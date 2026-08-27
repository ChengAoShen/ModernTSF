---
name: "Informer"
implementation: upstream
summary: "Informer is a Transformer-based model for long-sequence time-series forecasting in the standard univariate and multivariate setting. It introduces ProbSparse self-attention to achieve O(L log L) time and memory complexity, a self-attention distilling mechanism that halves cascading layer inputs to handle extreme-length inputs, and a generative-style decoder that produces the entire output sequence in a single forward pass, dramatically reducing inference latency on long-horizon tasks."
paper:
  title: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
  venue: "AAAI 2021"
  year: 2021
  url: "https://doi.org/10.1609/aaai.v35i12.17325"
codebase:
  url: "https://github.com/thuml/Time-Series-Library"
  revision: "2fb5b84ecef67c45a759f7cf82023d27afe27882"
  license: "MIT"
  usage: ported
---
# Informer

Informer is a Transformer-based model for long-sequence time-series forecasting in the standard univariate and multivariate setting. It introduces ProbSparse self-attention to achieve O(L log L) time and memory complexity, a self-attention distilling mechanism that halves cascading layer inputs to handle extreme-length inputs, and a generative-style decoder that produces the entire output sequence in a single forward pass, dramatically reducing inference latency on long-horizon tasks.

## Paper
- **Title**: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- **Venue**: AAAI 2021
- **Published**: 2021 (arXiv: 2020-12)
- **arXiv**: https://arxiv.org/abs/2012.07436

## Abstract
Many real-world applications require the prediction of long sequence time-series, such as electricity consumption planning. Long sequence time-series forecasting (LSTF) demands a high prediction capacity of the model, which is the ability to capture precise long-range dependency coupling between output and input efficiently. Recent studies have shown the potential of Transformer to increase the prediction capacity. However, there are several severe issues with Transformer that prevent it from being directly applicable to LSTF, including quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. To address these issues, we design an efficient transformer-based model for LSTF, named Informer, with three distinctive characteristics: (i) a ProbSparse self-attention mechanism, which achieves O(L log L) in time complexity and memory usage, and has comparable performance on sequences' dependency alignment. (ii) the self-attention distilling highlights dominating attention by halving cascading layer input, and efficiently handles extreme long input sequences. (iii) the generative style decoder, while conceptually simple, predicts the long time-series sequences at one forward operation rather than a step-by-step way, which drastically improves the inference speed of long-sequence predictions. Extensive experiments on four large-scale datasets demonstrate that Informer significantly outperforms existing methods and provides a new solution to the LSTF problem.

## In ModernTSF
Default config: `configs/models/Informer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

Implementation: **upstream** (numerical parity pending). The forecasting implementation is pinned to
[`thuml/Time-Series-Library`](https://github.com/thuml/Time-Series-Library)
revision `2fb5b84ecef67c45a759f7cf82023d27afe27882` under MIT and traces to the
authors' Informer implementation. ProbSparse attention, encoder distillation,
the generative decoder, and temporal embeddings are retained through shared
components. ModernTSF keeps only long-term forecasting, constructs decoder
inputs in the common runner, and uses a smaller display preset with
`label_len=0`; it does not claim the published benchmark numbers.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZhouZPZLXZ21,
  author       = {Haoyi Zhou and
                  Shanghang Zhang and
                  Jieqi Peng and
                  Shuai Zhang and
                  Jianxin Li and
                  Hui Xiong and
                  Wancai Zhang},
  title        = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series
                  Forecasting},
  booktitle    = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2021, Thirty-Third Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
                  2021},
  pages        = {11106--11115},
  publisher    = {{AAAI} Press},
  year         = {2021},
  url          = {https://doi.org/10.1609/aaai.v35i12.17325},
  doi          = {10.1609/AAAI.V35I12.17325},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/ZhouZPZLXZ21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
