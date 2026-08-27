---
model: "QuantilePatchTST"
forecasting_setting: "time_series"
output_type: "quantile"
config: "configs/models/QuantilePatchTST.toml"
spec: "models.quantile_patchtst.spec"
paper_title: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST backbone)"
venue: "ICLR 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2211.14730"
---
# QuantilePatchTST

QuantilePatchTST is a **probabilistic** ModernTSF forecaster: it wraps the
patch-based Transformer backbone PatchTST with the shared monotone `QuantileHead`
(`src/components/quantile_head.py`) to emit a non-crossing quantile grid
`(B, pred_len, C, Q)`. Quantiles are built from a median anchor via cumulative
`softplus` offsets, so they cannot cross. Trained with the pinball (`quantile`)
loss and scored with CRPS / WQL / coverage.

## Method
- **Backbone**: PatchTST — channel-independent patching + Transformer encoder
  (Nie et al., ICLR 2023, arXiv: 2211.14730).
- **Probabilistic head**: monotone quantile regression (pinball loss).

## In ModernTSF
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/QuantilePatchTST.toml`; specification: `spec.py`; adapter:
`model.py`. `quantile_levels` are injected from
`evaluation.quantile_levels`. Use the model specification and probabilistic output contract.

## Source and verification

- Evidence: `adaptation` using `yuqinie98/PatchTST` revision `204c21efe0b39603ad6e2ca640ef5896646ab1a9` (Apache-2.0).
- The local PatchTST backbone is composed with ModernTSF's monotone quantile head. The cited paper's point-forecast results do not validate this probabilistic composition.
