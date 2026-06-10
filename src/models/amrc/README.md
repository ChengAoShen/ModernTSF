---
model: "AMRC"
forecasting_setting: "time_series"
config: "configs/models/AMRC.toml"
registry: "models.amrc.registry"
paper_title: "Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency"
venue: "NeurIPS 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2510.19980"
---
# AMRC

AMRC (Adaptive Masking Loss with Representation Consistency) is a plug-and-play training framework for time-series forecasting that addresses redundant feature learning. Rather than introducing a new architecture, it wraps any existing forecasting model with a dynamic masking loss that adaptively identifies highly discriminative temporal segments and a representation consistency constraint that stabilises the mapping among inputs, labels, and predictions — serving the standard multivariate time-series forecasting setting.

## Paper
- **Title**: Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.19980

## Abstract
Time series forecasting plays a pivotal role in critical domains such as energy management and financial markets. Although deep learning-based approaches (e.g., MLP, RNN, Transformer) have achieved remarkable progress, the prevailing "long-sequence information gain hypothesis" exhibits inherent limitations. Through systematic experimentation, this study reveals a counterintuitive phenomenon: appropriately truncating historical data can paradoxically enhance prediction accuracy, indicating that existing models learn substantial redundant features (e.g., noise or irrelevant fluctuations) during training, thereby compromising effective signal extraction. Building upon information bottleneck theory, we propose an innovative solution termed Adaptive Masking Loss with Representation Consistency (AMRC), which features two core components: 1) Dynamic masking loss, which adaptively identified highly discriminative temporal segments to guide gradient descent during model training; 2) Representation consistency constraint, which stabilized the mapping relationships among inputs, labels, and predictions. Experimental results demonstrate that AMRC effectively suppresses redundant feature learning while significantly improving model performance. This work not only challenges conventional assumptions in temporal modeling but also provides novel theoretical insights and methodological breakthroughs for developing efficient and robust forecasting models.

## In ModernTSF
Default config: `configs/models/AMRC.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2510-19980,
  author       = {Renzhao Liang and
                  Sizhe Xu and
                  Chenggang Xie and
                  Jingru Chen and
                  Feiyang Ren and
                  Shu Yang and
                  Takahiro Yabe},
  title        = {Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking
                  Loss with Representation Consistency},
  journal      = {CoRR},
  volume       = {abs/2510.19980},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2510.19980},
  doi          = {10.48550/ARXIV.2510.19980},
  eprinttype   = {arXiv},
  eprint       = {2510.19980},
  timestamp    = {Wed, 04 Mar 2026 19:44:06 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2510-19980.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
