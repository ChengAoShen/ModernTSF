---
model: "AirPhyNet"
forecasting_setting: "covariate"
config: "configs/models/AirPhyNet.toml"
registry: "models.airphynet.registry"
paper_title: "AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction"
venue: "ICLR 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2402.03784"
---
# AirPhyNet

AirPhyNet is a physics-guided neural network for air quality prediction in the covariate prediction setting. It encodes two established physical principles of air particle movement — diffusion and advection — as differential equation networks, then integrates this physics knowledge through a graph structure to capture spatio-temporal relationships between monitoring stations using both historical target values and future covariates (requires `torchdiffeq`).

## Paper
- **Title**: AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.03784

## Abstract
Air quality prediction and modelling plays a pivotal role in public health and environment management, for individuals and authorities to make informed decisions. Although traditional data-driven models have shown promise in this domain, their long-term prediction accuracy can be limited, especially in scenarios with sparse or incomplete data and they often rely on black-box deep learning structures that lack solid physical foundation leading to reduced transparency and interpretability in predictions. To address these limitations, this paper presents a novel approach named Physics guided Neural Network for Air Quality Prediction (AirPhyNet). Specifically, we leverage two well-established physics principles of air particle movement (diffusion and advection) by representing them as differential equation networks. Then, we utilize a graph structure to integrate physics knowledge into a neural network architecture and exploit latent representations to capture spatio-temporal relationships within the air quality data. Experiments on two real-world benchmark datasets demonstrate that AirPhyNet outperforms state-of-the-art models for different testing scenarios including different lead time (24h, 48h, 72h), sparse data and sudden change prediction, achieving reduction in prediction errors up to 10%. Moreover, a case study further validates that our model captures underlying physical processes of particle movement and generates accurate predictions with real physical meaning.

## In ModernTSF
Default config: `configs/models/AirPhyNet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/HettigeJXLCW24,
  author       = {Kethmi Hirushini Hettige and
                  Jiahao Ji and
                  Shili Xiang and
                  Cheng Long and
                  Gao Cong and
                  Jingyuan Wang},
  title        = {AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality
                  Prediction},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=JW3jTjaaAB},
  timestamp    = {Mon, 13 Jan 2025 16:16:40 +0100},
  biburl       = {https://dblp.org/rec/conf/iclr/HettigeJXLCW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
