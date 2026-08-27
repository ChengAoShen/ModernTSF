---
name: "Aurora"
implementation: rewrite
summary: "Aurora is a Multimodal Time Series Foundation Model designed for universal generative forecasting across domains. It supports multimodal inputs (text and image alongside temporal data) and zero-shot cross-domain inference, serving the standard time series forecasting setting with both deterministic and probabilistic outputs."
paper:
  title: "Aurora: Towards Universal Generative Multimodal Time Series Forecasting"
  venue: "ICLR 2026"
  year: 2026
  url: "https://arxiv.org/abs/2509.22295"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# Aurora

Aurora is a Multimodal Time Series Foundation Model designed for universal generative forecasting across domains. It supports multimodal inputs (text and image alongside temporal data) and zero-shot cross-domain inference, serving the standard time series forecasting setting with both deterministic and probabilistic outputs.

<!-- model-card:canonical:start -->
## Method overview

Aurora is a Multimodal Time Series Foundation Model designed for universal generative forecasting across domains.

## Core architecture

It supports multimodal inputs (text and image alongside temporal data) and zero-shot cross-domain inference, serving the standard time series forecasting setting with both deterministic and probabilistic outputs.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.22295); title: Aurora: Towards Universal Generative Multimodal Time Series Forecasting; venue/year: ICLR 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/Aurora.toml`](../../../configs/models/Aurora.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Aurora: Towards Universal Generative Multimodal Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-09)
- **arXiv**: https://arxiv.org/abs/2509.22295

## Abstract
Cross-domain generalization is very important in Time Series Forecasting because similar historical information may lead to distinct future trends due to the domain-specific characteristics. Recent works focus on building unimodal time series foundation models and end-to-end multimodal supervised models. Since domain-specific knowledge is often contained in modalities like texts, the former lacks the explicit utilization of them, thus hindering the performance. The latter is tailored for end-to-end scenarios and does not support zero-shot inference for cross-domain scenarios. In this work, we introduce Aurora, a Multimodal Time Series Foundation Model, which supports multimodal inputs and zero-shot inference. Pretrained on Cross-domain Multimodal Time Series Corpus, Aurora can adaptively extract and focus on key domain knowledge contained in corresponding text or image modalities, thus possessing strong cross-domain generalization capability. Through tokenization, encoding, and distillation, Aurora can extract multimodal domain knowledge as guidance and then utilizes a Modality-Guided Multi-head Self-Attention to inject them into the modeling of temporal representations. In the decoding phase, the multimodal representations are used to generate the conditions and prototypes of future tokens, contributing to a novel Prototype-Guided Flow Matching for generative probabilistic forecasting. Comprehensive experiments on 5 well-recognized benchmarks, including TimeMMD, TSFM-Bench, ProbTS, TFB, and EPF, demonstrate the consistent state-of-the-art performance of Aurora on both unimodal and multimodal scenarios.

## In ModernTSF
Default config: `configs/models/Aurora.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2509-22295,
  author       = {Xingjian Wu and
                  Jianxin Jin and
                  Wanghui Qiu and
                  Peng Chen and
                  Yang Shu and
                  Bin Yang and
                  Chenjuan Guo},
  title        = {Aurora: Towards Universal Generative Multimodal Time Series Forecasting},
  journal      = {CoRR},
  volume       = {abs/2509.22295},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2509.22295},
  doi          = {10.48550/ARXIV.2509.22295},
  eprinttype   = {arXiv},
  eprint       = {2509.22295},
  timestamp    = {Mon, 20 Oct 2025 21:29:14 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2509-22295.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
