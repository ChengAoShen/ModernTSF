---
model: "QuantileDLinear"
forecasting_setting: "time_series"
output_type: "quantile"
config: "configs/models/QuantileDLinear.toml"
spec: "models.quantile_dlinear.spec"
paper_title: "Are Transformers Effective for Time Series Forecasting? (DLinear backbone)"
venue: "AAAI 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2205.13504"
---
# QuantileDLinear

QuantileDLinear is a **probabilistic** ModernTSF forecaster: it wraps the point
DLinear backbone with the shared monotone `QuantileHead`
(`src/components/quantile_head.py`) to emit a non-crossing grid of quantiles
`(B, pred_len, C, Q)` instead of a single point. The head builds quantiles from a
median anchor by adding/subtracting cumulative `softplus` offsets, so the
predicted quantiles cannot cross by construction. It is trained with the pinball
(`quantile`) loss and scored with CRPS / WQL / coverage.

## Method
- **Backbone**: DLinear — trend + seasonal decomposition with two linear heads
  (Zeng et al., AAAI 2023, arXiv: 2205.13504).
- **Probabilistic head**: monotone quantile regression (pinball loss; Koenker &
  Bassett, 1978).

## In ModernTSF
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/QuantileDLinear.toml`; specification: `spec.py`; adapter:
`model.py`. `quantile_levels` are injected from
`evaluation.quantile_levels`. Use the model specification and probabilistic output contract.
