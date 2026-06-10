---
model: "AirDualODE"
forecasting_setting: "covariate"
config: "configs/models/AirDualODE.toml"
registry: "models.airdualode.registry"
paper_title: "Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems"
venue: "ICLR 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2410.19892"
---
# AirDualODE

AirDualODE is a covariate prediction model designed for air quality forecasting in open atmospheric systems. It integrates dual branches of Neural ODEs — one grounded in physics-based open-system equations (a discrete Boundary-Aware Diffusion-Advection formulation) and one fully data-driven — whose representations are temporally aligned and fused to improve pollutant concentration prediction at node level. The model operates in the covariate forecasting setting, leveraging both historical target values and auxiliary covariates (including optionally known future covariates) and requires the `torchdiffeq` library.

## Paper
- **Title**: Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems
- **Venue**: ICLR 2025
- **Published**: 2025 (arXiv: 2024-10)
- **arXiv**: https://arxiv.org/abs/2410.19892

## Abstract
Air pollution significantly threatens human health and ecosystems, necessitating effective air quality prediction to inform public policy. Traditional approaches are generally categorized into physics-based and data-driven models. Physics-based models usually struggle with high computational demands and closed-system assumptions, while data-driven models may overlook essential physical dynamics, confusing the capturing of spatiotemporal correlations. Although some physics-guided approaches combine the strengths of both models, they often face a mismatch between explicit physical equations and implicit learned representations. To address these challenges, we propose Air-DualODE, a novel physics-guided approach that integrates dual branches of Neural ODEs for air quality prediction. The first branch applies open-system physical equations to capture spatiotemporal dependencies for learning physics dynamics, while the second branch identifies the dependencies not addressed by the first in a fully data-driven way. These dual representations are temporally aligned and fused to enhance prediction accuracy. Our experimental results demonstrate that Air-DualODE achieves state-of-the-art performance in predicting pollutant concentrations across various spatial scales, thereby offering a promising solution for real-world air quality challenges.

## In ModernTSF
Default config: `configs/models/AirDualODE.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/TianL0CGZPRY25,
  author       = {Jindong Tian and
                  Yuxuan Liang and
                  Ronghui Xu and
                  Peng Chen and
                  Chenjuan Guo and
                  Aoying Zhou and
                  Lujia Pan and
                  Zhongwen Rao and
                  Bin Yang},
  title        = {Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open
                  Systems},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025, Singapore, April 24-28, 2025},
  publisher    = {OpenReview.net},
  year         = {2025},
  url          = {https://openreview.net/forum?id=kOJf7Dklyv},
  timestamp    = {Fri, 14 Nov 2025 07:30:22 +0100},
  biburl       = {https://dblp.org/rec/conf/iclr/TianL0CGZPRY25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
