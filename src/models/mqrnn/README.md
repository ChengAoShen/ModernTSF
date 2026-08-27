---
model: "MQRNN"
forecasting_setting: "time_series"
output_type: "quantile"
config: "configs/models/MQRNN.toml"
spec: "models.mqrnn.spec"
paper_title: "A Multi-Horizon Quantile Recurrent Forecaster"
venue: "NeurIPS 2017 Time Series Workshop"
year: 2017
arxiv: "https://arxiv.org/abs/1711.11053"
---
# MQRNN

MQRNN (Multi-horizon Quantile Recurrent forecaster) is a **probabilistic**
sequence-to-sequence model: an RNN encoder summarizes the input window into a
context, and a global MLP decoder emits all horizon steps jointly as quantiles.
In ModernTSF the decoder feeds the shared monotone `QuantileHead`
(`src/components/quantile_head.py`), giving a non-crossing quantile grid
`(B, pred_len, C, Q)` trained with the pinball (`quantile`) loss and scored with
CRPS / WQL / coverage.

## Paper
- **Title**: A Multi-Horizon Quantile Recurrent Forecaster
- **Authors**: Wen, Torkkola, Narayanaswamy, Madeka
- **Published**: 2017
- **arXiv**: https://arxiv.org/abs/1711.11053

## In ModernTSF
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/MQRNN.toml`; specification: `spec.py`; implementation:
`model.py`. `quantile_levels` are injected from
`evaluation.quantile_levels`. Use the model specification and probabilistic output contract.

## Source and verification

- Evidence: `adaptation`; no author implementation or pinned upstream source has been established.
- This uses a shared channel-independent GRU, joint horizon MLP, and ModernTSF monotone quantile head. It does not implement the paper's static/future-covariate global/local decoder.
- Paper protocol and result reproduction remain blocked pending a traceable reference and dataset-aligned experiment.
