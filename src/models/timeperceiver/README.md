---
name: "TimePerceiver"
implementation: rewrite
summary: "TimePerceiver is a time series forecasting model built around a Perceiver-style encoder-decoder architecture. It generalises the forecasting task to arbitrary temporal prediction objectives (extrapolation, interpolation, and imputation) by dividing sequences into patch tokens, encoding them through a set of latent bottleneck representations that interact with all input patches via cross-attention to capture both temporal and cross-channel dependencies, and decoding future patches with learnable queries corresponding to target timestamps. The design is paired with a unified training strategy that tightly aligns the encoder, decoder, and prediction objectives."
paper:
  title: "TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting"
  venue: "NeurIPS 2025"
  year: 2025
  url: "https://arxiv.org/abs/2512.22550"
codebase:
  url: "https://github.com/efficient-learning-lab/TimePerceiver"
  revision: "7e30cc07b51c709f408409fd60a34c81ae8990be"
  license: "MIT"
  usage: reference-only
---
# TimePerceiver

TimePerceiver is a time series forecasting model built around a Perceiver-style encoder-decoder architecture. It generalises the forecasting task to arbitrary temporal prediction objectives (extrapolation, interpolation, and imputation) by dividing sequences into patch tokens, encoding them through a set of latent bottleneck representations that interact with all input patches via cross-attention to capture both temporal and cross-channel dependencies, and decoding future patches with learnable queries corresponding to target timestamps. The design is paired with a unified training strategy that tightly aligns the encoder, decoder, and prediction objectives.

## Paper
- **Title**: TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2024-12)
- **arXiv**: https://arxiv.org/abs/2512.22550

## Abstract
In machine learning, effective modeling requires a holistic consideration of how to encode inputs, make predictions (i.e., decoding), and train the model. However, in time-series forecasting, prior work has predominantly focused on encoder design, often treating prediction and training as separate or secondary concerns. In this paper, we propose TimePerceiver, a unified encoder-decoder forecasting framework that is tightly aligned with an effective training strategy. To be specific, we first generalize the forecasting task to include diverse temporal prediction objectives such as extrapolation, interpolation, and imputation. Since this generalization requires handling input and target segments that are arbitrarily positioned along the temporal axis, we design a novel encoder-decoder architecture that can flexibly perceive and adapt to these varying positions. For encoding, we introduce a set of latent bottleneck representations that can interact with all input segments to jointly capture temporal and cross-channel dependencies. For decoding, we leverage learnable queries corresponding to target timestamps to effectively retrieve relevant information. Extensive experiments demonstrate that our framework consistently and significantly outperforms prior state-of-the-art baselines across a wide range of benchmark datasets. The code is available at https://github.com/efficient-learning-lab/TimePerceiver.

## In ModernTSF
Default config: `configs/models/TimePerceiver.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

Compared with the MIT-licensed author repository at `7e30cc07b51c709f408409fd60a34c81ae8990be`. The encoder/latent/query decoder is retained, while generalized arbitrary-index training is reduced to ordinary past-to-future forecasting. Implementation: `rewrite` (clean-room audit pending).

## Citation

```bibtex
@misc{lee2025timeperceiver,
  author        = {Jaebin Lee and
                  Hankook Lee},
  title         = {TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting},
  year          = {2025},
  eprint        = {2512.22550},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2512.22550}
}
```
