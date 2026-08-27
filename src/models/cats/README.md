---
model: "CATS"
forecasting_setting: "time_series"
config: "configs/models/CATS.toml"
spec: "models.cats.spec"
paper_title: "Are Self-Attentions Effective for Time Series Forecasting?"
venue: "NeurIPS 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2405.16877"
---
# CATS

CATS (Cross-Attention-only Time Series transformer) is a multivariate time series forecasting model that eliminates self-attention entirely from the Transformer architecture and relies solely on cross-attention mechanisms, using future horizon-dependent parameters as queries with enhanced parameter sharing to improve long-term forecasting accuracy while reducing parameter count and memory usage.

## Paper
- **Title**: Are Self-Attentions Effective for Time Series Forecasting?
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.16877

## Abstract
Time series forecasting is crucial for applications across multiple domains and various scenarios. Although Transformer models have dramatically advanced the landscape of forecasting, their effectiveness remains debated. Recent findings have indicated that simpler linear models might outperform complex Transformer-based approaches, highlighting the potential for more streamlined architectures. In this paper, we shift the focus from evaluating the overall Transformer architecture to specifically examining the effectiveness of self-attention for time series forecasting. To this end, we introduce a new architecture, Cross-Attention-only Time Series transformer (CATS), that rethinks the traditional Transformer framework by eliminating self-attention and leveraging cross-attention mechanisms instead. By establishing future horizon-dependent parameters as queries and enhanced parameter sharing, our model not only improves long-term forecasting accuracy but also reduces the number of parameters and memory usage. Extensive experiment across various datasets demonstrates that our model achieves superior performance with the lowest mean squared error and uses fewer parameters compared to existing models.

## In ModernTSF
Default config: `configs/models/CATS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

- **Paper**: the NeurIPS 2024 OpenReview record and the authors' official repository identify the same CATS architecture.
- **Code basis**: `dongbeank/CATS`, pinned to `58854fc759d608ce400f378be83f4513960e505d`, MIT license; the defining implementation is `models/CATS.py`.
- **Evidence**: `upstream-port`. The port preserves patch extraction, horizon-dependent dummy queries, cross-attention-only decoding, query-adaptive masking, normalization, and horizon projection.
- **Runtime differences**: argument parsing and tensor permutation are replaced by the shared model signature. The official experiment uses MSE; this repository's runner owns the selected training objective. `patch_len`, stride, attention dropout, query independence, padding, and attention storage remain explicit parameters.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/Kim00K24,
  author       = {Dongbin Kim and
                  Jinseong Park and
                  Jaewook Lee and
                  Hoki Kim},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {Are Self-Attentions Effective for Time Series Forecasting?},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/cf66f995883298c4db2f0dcba28fb211-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/Kim00K24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
