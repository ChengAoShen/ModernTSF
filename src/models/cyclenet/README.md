---
model: "CycleNet"
forecasting_setting: "time_series"
config: "configs/models/CycleNet.toml"
registry: "models.cyclenet.registry"
paper_title: "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns"
venue: "NeurIPS 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2409.18479"
---
# CycleNet

CycleNet is a long-term time-series forecasting model that explicitly models periodic patterns in the input sequence via a Residual Cycle Forecasting (RCF) technique. It separates learnable recurrent cycle components from the residual signal and predicts on the residuals, achieving state-of-the-art accuracy in electricity, weather, and energy domains with over 90% fewer parameters than competing approaches.

## Paper
- **Title**: CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns
- **Venue**: NeurIPS 2024 (Spotlight)
- **Published**: 2024 (arXiv: 2024-09)
- **arXiv**: https://arxiv.org/abs/2409.18479

## Abstract
The stable periodic patterns present in time series data serve as the foundation for conducting long-horizon forecasts. In this paper, we pioneer the exploration of explicitly modeling this periodicity to enhance the performance of models in long-term time series forecasting (LTSF) tasks. Specifically, we introduce the Residual Cycle Forecasting (RCF) technique, which utilizes learnable recurrent cycles to model the inherent periodic patterns within sequences, and then performs predictions on the residual components of the modeled cycles. Combining RCF with a Linear layer or a shallow MLP forms the simple yet powerful method proposed in this paper, called CycleNet. CycleNet achieves state-of-the-art prediction accuracy in multiple domains including electricity, weather, and energy, while offering significant efficiency advantages by reducing over 90% of the required parameter quantity. Furthermore, as a novel plug-and-play technique, the RCF can also significantly improve the prediction accuracy of existing models, including PatchTST and iTransformer. The source code is available at: https://github.com/ACAT-SCUT/CycleNet.

## In ModernTSF
Default config: `configs/models/CycleNet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/Lin0HWMZ24,
  author       = {Shengsheng Lin and
                  Weiwei Lin and
                  Xinyi Hu and
                  Wentai Wu and
                  Ruichao Mo and
                  Haocheng Zhong},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {CycleNet: Enhancing Time Series Forecasting through Modeling Periodic
                  Patterns},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/bfe7998398779dde03cad7a73b1f81b6-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/Lin0HWMZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
