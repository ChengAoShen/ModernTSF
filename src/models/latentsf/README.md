---
model: "LatentTSF"
forecasting_setting: "time_series"
config: "configs/models/LatentTSF.toml"
registry: "models.latentsf.registry"
paper_title: "From Observations to States: Latent Time Series Forecasting"
venue: "ICML 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2602.00297"
---
# LatentTSF

LatentTSF is a time series forecasting model that shifts the forecasting paradigm from observation-space regression to latent state prediction. It employs an AutoEncoder to project each observation into a learned higher-dimensional latent state space, then performs all forecasting entirely within that space, allowing the model to capture structured temporal dynamics rather than fitting noisy observations directly. This addresses the "Latent Chaos" phenomenon where standard observation-space models achieve accurate predictions while learning temporally disordered representations.

## Paper
- **Title**: From Observations to States: Latent Time Series Forecasting
- **Venue**: ICML 2026
- **Published**: 2026 (arXiv: 2026-01)
- **arXiv**: https://arxiv.org/abs/2602.00297

## Abstract
Deep learning has achieved strong performance in Time Series Forecasting (TSF). However, we identify a critical representation paradox, termed Latent Chaos: models with accurate predictions often learn latent representations that are temporally disordered and lack continuity. We attribute this to the dominant observation-space forecasting paradigm, where minimizing point-wise errors on noisy and partially observed data encourages shortcut solutions instead of the recovery of underlying system dynamics. To address this, we propose Latent Time Series Forecasting (LatentTSF), a paradigm that shifts TSF from observation regression to latent state prediction. LatentTSF employs an AutoEncoder to project each observation into a learned latent state space and performs forecasting entirely in this space, allowing the model to focus on learning structured temporal dynamics. We provide an information-theoretic analysis showing that the latent objectives can be motivated as surrogates for maximizing mutual information between predicted and ground-truth latent states and future observations. Extensive experiments on widely-used benchmarks confirm that LatentTSF effectively mitigates latent chaos, yielding consistent improvements in both forecasting accuracy and representation quality.

## In ModernTSF
Default config: `configs/models/LatentTSF.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2602-00297,
  author       = {Jie Yang and
                  Yifan Hu and
                  Yuante Li and
                  Kexin Zhang and
                  Kaize Ding and
                  Philip S. Yu},
  title        = {From Observations to States: Latent Time Series Forecasting},
  journal      = {CoRR},
  volume       = {abs/2602.00297},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2602.00297},
  doi          = {10.48550/ARXIV.2602.00297},
  eprinttype   = {arXiv},
  eprint       = {2602.00297},
  timestamp    = {Thu, 12 Mar 2026 08:05:41 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2602-00297.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
