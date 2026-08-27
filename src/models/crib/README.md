---
model: "CRIB"
forecasting_setting: "time_series"
config: "configs/models/CRIB.toml"
spec: "models.crib.spec"
paper_title: "CRIB: Consistency-Regularized Information Bottleneck for Multivariate Time Series Forecasting with Missing Values"
venue: "to confirm"
year: null
arxiv: ""
upstream: "https://github.com/Muyiiiii/CRIB"
license: "to confirm"
---
# CRIB

CRIB is a forecasting port of a missing-value TSF architecture. In ModernTSF it
trains on complete standard forecasting windows (the upstream missing-value data
pipeline is not included). It patches the input, encodes it with a TCN +
unified-variate Transformer into an Information-Bottleneck latent, and predicts
with a small MLP head. A consistency regularizer aligns the representations of
the clean input and a noisy second view, while an IB (KL) term compresses the
latent — together filtering the noise that missing values inject.

## Training objective
`L = IB_weight · MAE(Ŷ, Y) + Consis_weight · MSE(enc_clean, enc_noisy) + KL_weight · KL(q(z|x)‖N(0,I))`
(defaults `IB_weight=1`, `Consis_weight=1`, `KL_weight=1e-6`).

## In ModernTSF
Default config: `configs/models/CRIB.toml`; specification: `spec.py`; implementation:
`model.py`.

**Model-only port** (per request): the upstream missing-value masking /
augmentation **data pipeline is NOT included** — CRIB trains on the standard
complete forecasting windows (equivalent to upstream `missing_rate=0`). The
vendored core reproduces the upstream architecture (a patching adapter maps the
`(B, seq_len, enc_in)` input to the patched 4-D tensor CRIB expects; dead/unused
upstream submodules are dropped). The consistency + KL terms are computed inside
`forward` from the input alone and exposed via the trainer's `aux_loss`
convention; the MAE prediction term is the configured `training.loss` (use
`mae`). Constraints: `patch_len` must divide `seq_len`, and `model_dim` must be
divisible by `heads_num`. Verify with
`uv run tsf smoke --model CRIB`.

Upstream reference: https://github.com/Muyiiiii/CRIB
