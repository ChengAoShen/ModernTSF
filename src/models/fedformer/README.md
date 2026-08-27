---
name: "FEDformer"
implementation: rewrite
summary: "FEDformer is a Transformer-based model for long-term multivariate and univariate time-series forecasting that combines seasonal-trend decomposition with a frequency-enhanced attention mechanism. The decomposition component captures the global profile of the series while Transformer blocks model finer-grained structure; exploiting the sparse Fourier representation of most time series yields linear complexity in sequence length, making FEDformer more efficient than standard Transformers."
paper:
  title: "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
  venue: "ICML 2022"
  year: 2022
  url: "https://proceedings.mlr.press/v162/zhou22g.html"
codebase:
  url: "https://github.com/MAZiqing/FEDformer"
  revision: "c0f6b972def125691434d62be1ecadf710ae921a"
  license: "MIT"
  usage: reference-only
---
# FEDformer

FEDformer is a Transformer-based model for long-term multivariate and univariate time-series forecasting that combines seasonal-trend decomposition with a frequency-enhanced attention mechanism. The decomposition component captures the global profile of the series while Transformer blocks model finer-grained structure; exploiting the sparse Fourier representation of most time series yields linear complexity in sequence length, making FEDformer more efficient than standard Transformers.

## Paper
- **Title**: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
- **Venue**: ICML 2022
- **Published**: 2022 (arXiv: 2022-01)
- **arXiv**: https://arxiv.org/abs/2201.12740

## Abstract
Although Transformer-based methods have significantly improved state-of-the-art results for long-term series forecasting, they are not only computationally expensive but more importantly, are unable to capture the global view of time series (e.g. overall trend). To address these problems, we propose to combine Transformer with the seasonal-trend decomposition method, in which the decomposition method captures the global profile of time series while Transformers capture more detailed structures. To further enhance the performance of Transformer for long-term prediction, we exploit the fact that most time series tend to have a sparse representation in well-known basis such as Fourier transform, and develop a frequency enhanced Transformer. Besides being more effective, the proposed method, termed as Frequency Enhanced Decomposed Transformer (FEDformer), is more efficient than standard Transformer with a linear complexity to the sequence length. Our empirical studies with six benchmark datasets show that compared with state-of-the-art methods, FEDformer can reduce prediction error by 14.8% and 22.6% for multivariate and univariate time series, respectively.

## In ModernTSF
Default config: `configs/models/FEDformer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

This entry is a forecast-only **modified integration** checked against the MIT-licensed
author revision `c0f6b972def125691434d62be1ecadf710ae921a` and THUML
Time-Series-Library revision `4e938a1767106324dd753b2a44832bf870a0252e`.
Only the Fourier variant is present. Its shared Fourier implementation follows
the later configurable-head, explicit real/imaginary form; calendar embeddings
consume the benchmark's six raw fields, and zero decoder context is handled
explicitly. Wavelet layers, upstream training, and published metric parity are
not included.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/ZhouMWW0022,
  author       = {Tian Zhou and
                  Ziqing Ma and
                  Qingsong Wen and
                  Xue Wang and
                  Liang Sun and
                  Rong Jin},
  editor       = {Kamalika Chaudhuri and
                  Stefanie Jegelka and
                  Le Song and
                  Csaba Szepesv{\'{a}}ri and
                  Gang Niu and
                  Sivan Sabato},
  title        = {FEDformer: Frequency Enhanced Decomposed Transformer for Long-term
                  Series Forecasting},
  booktitle    = {International Conference on Machine Learning, {ICML} 2022, 17-23 July
                  2022, Baltimore, Maryland, {USA}},
  series       = {Proceedings of Machine Learning Research},
  pages        = {27268--27286},
  publisher    = {{PMLR}},
  year         = {2022},
  url          = {https://proceedings.mlr.press/v162/zhou22g.html},
  timestamp    = {Thu, 23 Jan 2025 19:51:39 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/ZhouMWW0022.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
