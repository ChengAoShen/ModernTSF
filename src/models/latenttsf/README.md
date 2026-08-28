---
name: "LatentTSF"
implementation: rewrite
summary: "LatentTSF is a time series forecasting model that shifts the forecasting paradigm from observation-space regression to latent state prediction. It employs an AutoEncoder to project each observation into a learned higher-dimensional latent state space, then performs all forecasting entirely within that space, allowing the model to capture structured temporal dynamics rather than fitting noisy observations directly. This addresses the \"Latent Chaos\" phenomenon where standard observation-space models achieve accurate predictions while learning temporally disordered representations."
paper:
  title: "From Observations to States: Latent Time Series Forecasting"
  venue: "ICML 2026"
  year: 2026
  url: "https://arxiv.org/abs/2602.00297"
codebase:
  url: "https://github.com/Muyiiiii/LatentTSF"
  revision: "7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2"
  license: "MIT"
  usage: reference-only
---
# LatentTSF

LatentTSF is a time series forecasting model that shifts the forecasting paradigm from observation-space regression to latent state prediction. It employs an AutoEncoder to project each observation into a learned higher-dimensional latent state space, then performs all forecasting entirely within that space, allowing the model to capture structured temporal dynamics rather than fitting noisy observations directly. This addresses the "Latent Chaos" phenomenon where standard observation-space models achieve accurate predictions while learning temporally disordered representations.

<!-- model-card:canonical:start -->
## Method overview

LatentTSF is a time series forecasting model that shifts the forecasting paradigm from observation-space regression to latent state prediction.

## Core architecture

It employs an AutoEncoder to project each observation into a learned higher-dimensional latent state space, then performs all forecasting entirely within that space, allowing the model to capture structured temporal dynamics rather than fitting noisy observations directly. This addresses the "Latent Chaos" phenomenon where standard observation-space models achieve accurate predictions while learning temporally disordered representations.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2602.00297); title: From Observations to States: Latent Time Series Forecasting; venue/year: ICML 2026 / 2026
- [codebase](https://github.com/Muyiiiii/LatentTSF); revision: `7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2`; license: `MIT`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/LatentTSF.toml`](../../../configs/models/LatentTSF.toml).

## Differences

Clean-room implementation: confirmed. Paper mapping: observation-to-state projection → `LatentStateAutoencoder`; latent forecasting → shared `DLinearBackbone`; Eq. 5 latent prediction/alignment objective → `train_loss_override`; two-stage freezing → `pretrain`. Reference-only source code was not copied. Optional perceptual loss, external checkpoints and numerical parity are not included.

## Shared components

- [`dlinear`](../../components/dlinear.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `d_ff=128`, `mse_weight=10.0`, `cosine_weight=15.0`, `use_latent_norm=True`, `kernel_size=25`, `individual=False`, `ae_train_epochs=100`, `ae_lr=0.0005`, `ae_loss='MAE'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: From Observations to States: Latent Time Series Forecasting
- **Venue**: ICML 2026
- **Published**: 2026 (arXiv: 2026-01)
- **arXiv**: https://arxiv.org/abs/2602.00297

## Abstract
Deep learning has achieved strong performance in Time Series Forecasting (TSF). However, we identify a critical representation paradox, termed Latent Chaos: models with accurate predictions often learn latent representations that are temporally disordered and lack continuity. We attribute this to the dominant observation-space forecasting paradigm, where minimizing point-wise errors on noisy and partially observed data encourages shortcut solutions instead of the recovery of underlying system dynamics. To address this, we propose Latent Time Series Forecasting (LatentTSF), a paradigm that shifts TSF from observation regression to latent state prediction. LatentTSF employs an AutoEncoder to project each observation into a learned latent state space and performs forecasting entirely in this space, allowing the model to focus on learning structured temporal dynamics. We provide an information-theoretic analysis showing that the latent objectives can be motivated as surrogates for maximizing mutual information between predicted and ground-truth latent states and future observations. Extensive experiments on widely-used benchmarks confirm that LatentTSF effectively mitigates latent chaos, yielding consistent improvements in both forecasting accuracy and representation quality.

## In ModernTSF
Default config: `configs/models/LatentTSF.toml`; model specification: `spec.py`; implementation: `model.py`.

This is an independent two-stage implementation of the public method:

- **Stage 1 (pretrain).** A per-timestep MLP autoencoder `(E, D)` is pretrained
  by reconstruction (`ae_loss="MAE"` by default), then **frozen**. `E` maps each
  timestep's `enc_in`-dim observation to a `d_model`-dim latent state. It runs
  once per run via a `pretrain(train_loader, device)` hook called by `run_one`
  (skip with `ae_train_epochs=0`).
- **Stage 2 (forecast).** A DLinear backbone (the paper's primary Table-1
  backbone) forecasts **entirely in the frozen latent space**:
  `X -E-> Z_X -f-> Ẑ_Y -D-> Ŷ`, with `Z_Y = E(Y)`. The training objective is
  latent-only (paper Eq. 5): `L = mse_weight·‖Z_Y-Ẑ_Y‖² + cosine_weight·(1-cos)`,
  defaults `mse_weight=10` (α), `cosine_weight=15` (β). The observation-space
  loss is **not** part of the default objective (the optional perceptual loss is
  off, matching Sec. 5.3.1); early-stopping still uses observation-space MSE.

The rewrite relies on three opt-in, no-op-for-other-models trainer conventions
(`requires_train_target`/`set_train_target`, `train_loss_override`, and the `pretrain` hook —
see `benchmark.runner.trainer` / `benchmark.runner.run_one`). Key params:
`d_model`, `d_ff`, `mse_weight`, `cosine_weight`, `use_latent_norm`,
`ae_train_epochs`, `ae_lr`, `ae_loss`, plus DLinear's `kernel_size`/`individual`.
Raise `ae_train_epochs` toward 500 for paper-faithful AE pretraining. Verify with
`uv run tsf smoke --model LatentTSF`.

## Source and verification

Clean-room implementation: confirmed. Paper mapping: observation-to-state projection → `LatentStateAutoencoder`; latent forecasting → shared `DLinearBackbone`; Eq. 5 latent prediction/alignment objective → `train_loss_override`; two-stage freezing → `pretrain`. Reference-only source code was not copied. Optional perceptual loss, external checkpoints and numerical parity are not included.

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
