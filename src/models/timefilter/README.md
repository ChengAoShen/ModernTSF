---
name: "TimeFilter"
implementation: rewrite
summary: "TimeFilter is a GNN-based model for multivariate time-series forecasting that performs adaptive, fine-grained dependency modelling at the patch level. It segments the input sequence into non-overlapping patches and constructs a spatial-temporal graph, then applies a Mixture-of-Experts dynamic router to filter irrelevant inter-channel correlations and an adaptive graph learning module to aggregate the most critical spatial-temporal dependencies, combining the complementary strengths of channel-independent and channel-dependent strategies without manual channel clustering."
paper:
  title: "TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting"
  venue: "ICML 2025"
  year: 2025
  url: "https://arxiv.org/abs/2501.13041"
codebase:
  url: "https://github.com/TROUBADOUR000/TimeFilter"
  revision: "dffde87e4fff0fdeeebbacde03dc1e432e15b3a1"
  license: ""
  usage: reference-only
---
# TimeFilter

TimeFilter is a GNN-based model for multivariate time-series forecasting that performs adaptive, fine-grained dependency modelling at the patch level. It segments the input sequence into non-overlapping patches and constructs a spatial-temporal graph, then applies a Mixture-of-Experts dynamic router to filter irrelevant inter-channel correlations and an adaptive graph learning module to aggregate the most critical spatial-temporal dependencies, combining the complementary strengths of channel-independent and channel-dependent strategies without manual channel clustering.

<!-- model-card:canonical:start -->
## Method overview

TimeFilter is a GNN-based model for multivariate time-series forecasting that performs adaptive, fine-grained dependency modelling at the patch level.

## Core architecture

It segments the input sequence into non-overlapping patches and constructs a spatial-temporal graph, then applies a Mixture-of-Experts dynamic router to filter irrelevant inter-channel correlations and an adaptive graph learning module to aggregate the most critical spatial-temporal dependencies, combining the complementary strengths of channel-independent and channel-dependent strategies without manual channel clustering.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2501.13041); title: TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting; venue/year: ICML 2025 / 2025
- [codebase](https://github.com/TROUBADOUR000/TimeFilter); revision: `dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`; license: `not available`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TimeFilter.toml`](../../../configs/models/TimeFilter.toml).

## Differences

Implementation: **rewrite** (clean-room audit pending), compared with `TROUBADOUR000/TimeFilter@dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`. Patch-specific graph construction, region filtration and expert routing are retained. The author repository has no license file, and the common trainer cannot consume the official auxiliary MoE loss; hard-routing parameters are consequently frozen and the loss remains inspectable as `last_moe_loss`.

## Shared components

- [`embed`](../../components/embed.py)
- [`revin`](../../components/revin.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `d_ff=128`, `n_heads=4`, `e_layers=2`, `patch_len=16`, `dropout=0.1`, `alpha=0.1`, `top_p=0.5`, `pos=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting
- **Venue**: ICML 2025
- **Published**: 2025 (arXiv: 2025-01)
- **arXiv**: https://arxiv.org/abs/2501.13041

## Abstract
Time series forecasting methods generally fall into two main categories: Channel Independent (CI) and Channel Dependent (CD) strategies. While CI overlooks important covariate relationships, CD captures all dependencies without distinction, introducing noise and reducing generalization. Recent advances in Channel Clustering (CC) aim to refine dependency modeling by grouping channels with similar characteristics and applying tailored modeling techniques. However, coarse-grained clustering struggles to capture complex, time-varying interactions effectively. To address these challenges, we propose TimeFilter, a GNN-based framework for adaptive and fine-grained dependency modeling. After constructing the graph from the input sequence, TimeFilter refines the learned spatial-temporal dependencies by filtering out irrelevant correlations while preserving the most critical ones in a patch-specific manner. Extensive experiments on 13 real-world datasets from diverse application domains demonstrate the state-of-the-art performance of TimeFilter. The code is available at https://github.com/TROUBADOUR000/TimeFilter.

## In ModernTSF
Default config: `configs/models/TimeFilter.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

Implementation: **rewrite** (clean-room audit pending), compared with `TROUBADOUR000/TimeFilter@dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`. Patch-specific graph construction, region filtration and expert routing are retained. The author repository has no license file, and the common trainer cannot consume the official auxiliary MoE loss; hard-routing parameters are consequently frozen and the loss remains inspectable as `last_moe_loss`.

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
