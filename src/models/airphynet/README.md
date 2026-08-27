---
model: "AirPhyNet"
forecasting_setting: "covariate"
config: "configs/models/AirPhyNet.toml"
spec: "models.airphynet.spec"
paper_title: "AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction"
venue: "ICLR 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2402.03784"
---
# AirPhyNet

The AirPhyNet paper represents diffusion and advection with differential-equation networks and graph structure for physically guided air-quality prediction. This ModernTSF entry is a diffusion-only approximation over historical target values and time marks; the official wind-driven advection path is absent. It requires `torchdiffeq`.

## Paper
- **Title**: AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.03784

## Abstract
Air quality prediction and modelling plays a pivotal role in public health and environment management, for individuals and authorities to make informed decisions. Although traditional data-driven models have shown promise in this domain, their long-term prediction accuracy can be limited, especially in scenarios with sparse or incomplete data and they often rely on black-box deep learning structures that lack solid physical foundation leading to reduced transparency and interpretability in predictions. To address these limitations, this paper presents a novel approach named Physics guided Neural Network for Air Quality Prediction (AirPhyNet). Specifically, we leverage two well-established physics principles of air particle movement (diffusion and advection) by representing them as differential equation networks. Then, we utilize a graph structure to integrate physics knowledge into a neural network architecture and exploit latent representations to capture spatio-temporal relationships within the air quality data. Experiments on two real-world benchmark datasets demonstrate that AirPhyNet outperforms state-of-the-art models for different testing scenarios including different lead time (24h, 48h, 72h), sparse data and sudden change prediction, achieving reduction in prediction errors up to 10%. Moreover, a case study further validates that our model captures underlying physical processes of particle movement and generates accurate predictions with real physical meaning.

## In ModernTSF
Default config: `configs/models/AirPhyNet.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

- Official source: https://github.com/kethmih/AirPhyNet at `e77576cfea777e8cd07f2ae198c560a8790f4b91` (MIT).
- Evidence: `unverified`. The implementation was consolidated from CauAir's baseline and has not been numerically compared with the pinned official code.
- Known differences: the ODE function omits the official wind-driven advection dynamics and its edge attributes; missing graph input falls back to an identity graph; it fixes three trajectory samples and uses benchmark-level training instead of the official supervisor and preprocessing.

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
