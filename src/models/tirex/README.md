---
model: "TiRex"
forecasting_setting: "time_series"
config: "configs/models/TiRex.toml"
registry: "models.tirex.registry"
paper_title: "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning"
venue: "NeurIPS 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2505.23719"
---
# TiRex

TiRex is a time series forecasting model built on the xLSTM architecture that enables zero-shot prediction through enhanced in-context learning. It treats past observed values as context for forecasting future values, employs a training-time Contiguous Patch Masking (CPM) strategy to strengthen long-horizon state tracking, and is registered in ModernTSF as a lightweight native adapter using the shared recent-TSF prediction interface and normalization path.

## Paper
- **Title**: TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.23719

## Abstract
In-context learning, the ability of large language models to perform tasks using only examples provided in the prompt, has recently been adapted for time series forecasting. This paradigm enables zero-shot prediction, where past values serve as context for forecasting future values, making powerful forecasting tools accessible to non-experts and increasing the performance when training data are scarce. Most existing zero-shot forecasting approaches rely on transformer architectures, which, despite their success in language, often fall short of expectations in time series forecasting, where recurrent models like LSTMs frequently have the edge. Conversely, while LSTMs are well-suited for time series modeling due to their state-tracking capabilities, they lack strong in-context learning abilities. We introduce TiRex that closes this gap by leveraging xLSTM, an enhanced LSTM with competitive in-context learning skills. Unlike transformers, state-space models, or parallelizable RNNs such as RWKV, TiRex retains state-tracking, a critical property for long-horizon forecasting. To further facilitate its state-tracking ability, we propose a training-time masking strategy called CPM. TiRex sets a new state of the art in zero-shot time series forecasting on the HuggingFace benchmarks GiftEval and Chronos-ZS, outperforming significantly larger models including TabPFN-TS (Prior Labs), Chronos Bolt (Amazon), TimesFM (Google), and Moirai (Salesforce) across both short- and long-term forecasts.

## In ModernTSF
Default config: `configs/models/TiRex.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2505-23719,
  author       = {Andreas Auer and
                  Patrick Podest and
                  Daniel Klotz and
                  Sebastian B{\"{o}}ck and
                  G{\"{u}}nter Klambauer and
                  Sepp Hochreiter},
  title        = {TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced
                  In-Context Learning},
  journal      = {CoRR},
  volume       = {abs/2505.23719},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2505.23719},
  doi          = {10.48550/ARXIV.2505.23719},
  eprinttype   = {arXiv},
  eprint       = {2505.23719},
  timestamp    = {Tue, 05 Aug 2025 22:46:04 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2505-23719.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
