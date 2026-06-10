---
model: "TSRAG"
forecasting_setting: "time_series"
config: "configs/models/TSRAG.toml"
registry: "models.tsrag.registry"
paper_title: "TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster"
venue: "NeurIPS 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2503.07649"
---
# TSRAG

TSRAG (TS-RAG) is a retrieval-augmented generation framework for zero-shot time-series forecasting built on top of pre-trained Time Series Foundation Models (TSFMs). It uses a pre-trained time-series encoder to retrieve semantically relevant segments from a dedicated knowledge base and then fuses them with the TSFM's internal representations via a learnable Adaptive Retrieval Mixer (ARM) module — enhancing generalisation and interpretability without requiring task-specific fine-tuning. The model targets the standard multivariate time-series forecasting setting.

## Paper
- **Title**: TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-03)
- **arXiv**: https://arxiv.org/abs/2503.07649

## Abstract
Large Language Models (LLMs) and Foundation Models (FMs) have recently become prevalent for time series forecasting tasks. While fine-tuning LLMs enables domain adaptation, they often struggle to generalize across diverse and unseen datasets. Moreover, existing Time Series Foundation Models (TSFMs) still face challenges in handling non-stationary dynamics and distribution shifts, largely due to the lack of effective mechanisms for adaptation. To this end, we present TS-RAG, a retrieval-augmented generation framework for time series forecasting that enhances the generalization and interpretability of TSFMs. Specifically, TS-RAG leverages pre-trained time series encoders to retrieve semantically relevant segments from a dedicated knowledge base, enriching the contextual representation of the input query. Furthermore, we propose an Adaptive Retrieval Mixer (ARM) module that dynamically fuses the retrieved patterns with the TSFM's internal representation, improving forecasting accuracy without requiring task-specific fine-tuning. Thorough empirical studies on seven public benchmark datasets demonstrate that TS-RAG achieves state-of-the-art zero-shot forecasting performance, outperforming the existing TSFMs by up to 6.84% across diverse domains while also providing desirable interpretability.

## In ModernTSF
Default config: `configs/models/TSRAG.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2503-07649,
  author       = {Kanghui Ning and
                  Zijie Pan and
                  Yu Liu and
                  Yushan Jiang and
                  James Y. Zhang and
                  Kashif Rasul and
                  Anderson Schneider and
                  Lintao Ma and
                  Yuriy Nevmyvaka and
                  Dongjin Song},
  title        = {{TS-RAG:} Retrieval-Augmented Generation based Time Series Foundation
                  Models are Stronger Zero-Shot Forecaster},
  journal      = {CoRR},
  volume       = {abs/2503.07649},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2503.07649},
  doi          = {10.48550/ARXIV.2503.07649},
  eprinttype   = {arXiv},
  eprint       = {2503.07649},
  timestamp    = {Sun, 13 Apr 2025 20:58:54 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2503-07649.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
