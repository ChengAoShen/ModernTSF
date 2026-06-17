---
model: "QuantilePatchTST"
forecasting_setting: "time_series"
output_type: "quantile"
config: "configs/models/QuantilePatchTST.toml"
registry: "models.quantile_patchtst.registry"
paper_title: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST backbone)"
venue: "ICLR 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2211.14730"
---
# QuantilePatchTST

QuantilePatchTST is a **probabilistic** ModernTSF forecaster: it wraps the
patch-based Transformer backbone PatchTST with the shared monotone `QuantileHead`
(`src/models/_quantile_head.py`) to emit a non-crossing quantile grid
`(B, pred_len, C, Q)`. Quantiles are built from a median anchor via cumulative
`softplus` offsets, so they cannot cross. Trained with the pinball (`quantile`)
loss and scored with CRPS / WQL / coverage.

## Method
- **Backbone**: PatchTST — channel-independent patching + Transformer encoder
  (Nie et al., ICLR 2023, arXiv: 2211.14730).
- **Probabilistic head**: monotone quantile regression (pinball loss).

## In ModernTSF
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/QuantilePatchTST.toml`; schema: `schema.py`; adapter:
`model.py`; registry: `registry.py`. `quantile_levels` are injected from
`evaluation.quantile_levels`. See the `probabilistic-forecasting` skill.
