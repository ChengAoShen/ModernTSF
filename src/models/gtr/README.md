---
model: "GTR"
forecasting_setting: "time_series"
config: "configs/models/GTR.toml"
registry: "models.gtr.registry"
paper_title: "Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval"
venue: "ICLR 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2602.10847"
---
# GTR

GTR (Global Temporal Retriever) is a lightweight, plug-and-play module for multivariate time series forecasting that extends any host model's temporal receptive field beyond the immediate input window by maintaining an adaptive global temporal embedding of the full cycle and dynamically retrieving and aligning relevant long-range historical segments with the current input, fusing them via 2D convolution and residual connections.

## Paper
- **Title**: Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2026-02)
- **arXiv**: https://arxiv.org/abs/2602.10847

## Abstract
Multivariate time series forecasting (MTSF) plays a vital role in numerous real-world applications, yet existing models remain constrained by their reliance on a limited historical context. This limitation prevents them from effectively capturing global periodic patterns that often span cycles significantly longer than the input horizon - despite such patterns carrying strong predictive signals. Naive solutions, such as extending the historical window, lead to severe drawbacks, including overfitting, prohibitive computational costs, and redundant information processing. To address these challenges, we introduce the Global Temporal Retriever (GTR), a lightweight and plug-and-play module designed to extend any forecasting model's temporal awareness beyond the immediate historical context. GTR maintains an adaptive global temporal embedding of the entire cycle and dynamically retrieves and aligns relevant global segments with the input sequence. By jointly modeling local and global dependencies through a 2D convolution and residual fusion, GTR effectively bridges short-term observations with long-term periodicity without altering the host model architecture. Extensive experiments on six real-world datasets demonstrate that GTR consistently delivers state-of-the-art performance across both short-term and long-term forecasting scenarios, while incurring minimal parameter and computational overhead. These results highlight GTR as an efficient and general solution for enhancing global periodicity modeling in MTSF tasks.

## In ModernTSF
Default config: `configs/models/GTR.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2602-10847,
  author       = {Fanpu Cao and
                  Lu Dai and
                  Jindong Han and
                  Hui Xiong},
  title        = {Enhancing Multivariate Time Series Forecasting with Global Temporal
                  Retrieval},
  journal      = {CoRR},
  volume       = {abs/2602.10847},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2602.10847},
  doi          = {10.48550/ARXIV.2602.10847},
  eprinttype   = {arXiv},
  eprint       = {2602.10847},
  timestamp    = {Sun, 29 Mar 2026 14:37:55 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2602-10847.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
