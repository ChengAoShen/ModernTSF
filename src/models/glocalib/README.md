---
model: "GlocalIB"
forecasting_setting: "time_series"
config: "configs/models/GlocalIB.toml"
registry: "models.glocalib.registry"
paper_title: "Glocal Information Bottleneck for Time Series Imputation"
venue: "NeurIPS 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2510.04910"
---
# GlocalIB

Glocal-IB is a plug-in regularizer that aligns the latent embeddings of two
views of a series through a global-local Information Bottleneck: a projector on
one branch is pulled toward a stop-gradient embedding of the other branch,
improving representation quality. It is originally a **time-series imputation**
method (masked view vs complete view).

## In ModernTSF
Default config: `configs/models/GlocalIB.toml`; schema: `schema.py`;
implementation: `model.py`; registry: `registry.py`.

**Forecasting port** (ModernTSF is forecasting-only, no missingness). The
alignment mechanism is kept faithful and the two views are adapted: the **clean
lookback** `x` is the anchor (it always exists, so it produces the forecast and
its embedding is the detached alignment target), and an **augmented copy**
`x_aug` (random temporal masking, training-only) is the corrupted view whose
projected embedding is pulled toward the anchor. The wrapper + alignment losses
(`cos_align`, `contrastive`) are vendored pure-PyTorch (no pypots/pygrinder).

Objective: `L = L_pred(Ŷ, Y) + align_weight · (1 − mean cos(proj(emb_aug), emb.detach()))`.
The alignment term needs only `x`, so it rides the trainer's `aux_loss`
convention; eval is a plain single forward. Key params: `d_model`,
`align_weight`, `mask_ratio`, `align_loss_type`. Verify with
`uv run python tool/tsf.py smoke --model GlocalIB`.

Upstream reference: https://github.com/Muyiiiii/NeurIPS-25-Glocal-IB
