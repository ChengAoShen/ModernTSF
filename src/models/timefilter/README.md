---
model: "TimeFilter"
forecasting_setting: "time_series"
config: "configs/models/TimeFilter.toml"
registry: "models.timefilter.registry"
paper_title: "TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting"
venue: "ICML 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2501.13041"
---
# TimeFilter

TimeFilter is a GNN-based model for multivariate time-series forecasting that performs adaptive, fine-grained dependency modelling at the patch level. It segments the input sequence into non-overlapping patches and constructs a spatial-temporal graph, then applies a Mixture-of-Experts dynamic router to filter irrelevant inter-channel correlations and an adaptive graph learning module to aggregate the most critical spatial-temporal dependencies, combining the complementary strengths of channel-independent and channel-dependent strategies without manual channel clustering.

## Paper
- **Title**: TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting
- **Venue**: ICML 2025
- **Published**: 2025 (arXiv: 2025-01)
- **arXiv**: https://arxiv.org/abs/2501.13041

## Abstract
Time series forecasting methods generally fall into two main categories: Channel Independent (CI) and Channel Dependent (CD) strategies. While CI overlooks important covariate relationships, CD captures all dependencies without distinction, introducing noise and reducing generalization. Recent advances in Channel Clustering (CC) aim to refine dependency modeling by grouping channels with similar characteristics and applying tailored modeling techniques. However, coarse-grained clustering struggles to capture complex, time-varying interactions effectively. To address these challenges, we propose TimeFilter, a GNN-based framework for adaptive and fine-grained dependency modeling. After constructing the graph from the input sequence, TimeFilter refines the learned spatial-temporal dependencies by filtering out irrelevant correlations while preserving the most critical ones in a patch-specific manner. Extensive experiments on 13 real-world datasets from diverse application domains demonstrate the state-of-the-art performance of TimeFilter. The code is available at https://github.com/TROUBADOUR000/TimeFilter.

## In ModernTSF
Default config: `configs/models/TimeFilter.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/HuZLLLC0XP25,
  author       = {Yifan Hu and
                  Guibin Zhang and
                  Peiyuan Liu and
                  Disen Lan and
                  Naiqi Li and
                  Dawei Cheng and
                  Tao Dai and
                  Shu{-}Tao Xia and
                  Shirui Pan},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time
                  Series Forecasting},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/hu25ac.html},
  timestamp    = {Wed, 25 Feb 2026 07:39:03 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/HuZLLLC0XP25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
