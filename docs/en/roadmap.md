# Scope

ModernTSF is a **forecasting-only** benchmark. It targets forecasting in three
data settings (`task.mode`, see [task-modes.md](task-modes.md)):

- `time_series` — classic multivariate forecasting `(B, T, C)`.
- `spatiotemporal` — node-structured forecasting with an adjacency matrix.
- `covariate` — spatiotemporal + future-known covariates.

All 115 models and every dataset, metric, loss, and evaluation path serve these
three forecasting settings. `task.mode` exposes only the three settings above, so
every reachable code path is forecasting. The multi-task `task_name` branches that
ship with some upstream TSLib-style models (Autoformer, FEDformer, TimesNet, TiDE,
SegRNN, CrossLinear, MoFo) have been stripped during porting — there is no
`task_name` model parameter, schema field, or non-forecast branch anywhere in the
codebase.

## Explicitly out of scope (not planned)

These are **different task types**, not forecasting, and are intentionally **not**
part of ModernTSF: **imputation**, **anomaly detection**, **classification**, and
**foundation-model pretraining** (zero-shot LLM forecasters / large pretraining
corpora). They would each need their own dataset format, task contract, and
evaluation protocol; adding them is out of scope.

## Adopted from the benchmark survey (done)

Non-model assets adopted from BasicTS / TSLib / TFB (all serving the three
forecasting settings): extra metrics (`corr`/`rse`/`wape`/`smape`, `mase`
opt-in), masked losses (for missing-value forecasting), adjacency-normalization
utilities + `adj_norm`, many CSV + traffic datasets, the pluggable
training-callback layer (curriculum / grad-clip / grad-accum / aux-loss), scaler
enhancements, fit/inference timing, aggregation fairness (`--null-threshold`),
the RollingForecast evaluation strategy, and the dataset-characteristics tool.
