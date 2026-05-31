# Roadmap — deferred / out-of-scope tasks

ModernTSF currently targets **forecasting** in three data settings
(`time_series`, `spatiotemporal`, `covariate` — see [task-modes.md](task-modes.md)).
The items below are net-new **task types** (not enhancements of the three
forecasting modes). They are intentionally deferred: each needs its own dataset
format and evaluation protocol that cannot be smoke-verified within the current
forecasting harness. Several building blocks already exist, listed per item.

## M4 short-term forecasting track (deferred)

The M4 competition track (univariate series collection, per-frequency seasonality,
no input scaling, SMAPE/MASE/OWA scored against a Naive2 baseline) is a separate
sub-paradigm from long-term multivariate forecasting.

- **Needs:** an M4 dataset class (per-frequency Train/Test CSVs from the M4
  release), a no-scaler univariate path, the Naive2 baseline, and OWA aggregation.
- **Already in place:** `smape` and `mase` metrics (`METRIC_NAME_MAP`); the
  rolling-forecast evaluator could host per-series scoring.
- **Why deferred:** requires the M4 dataset download and a univariate-collection
  loader + protocol distinct from the windowed multivariate contract.

## Dedicated imputation task mode (deferred)

Masked imputation (mask observed timesteps, reconstruct them, score on masked
positions only) is a different task from forecasting.

- **Already in place:** the masked losses `masked_mae` / `masked_mse` /
  `masked_rmse` (with a `targets_mask`, BasicTS convention) — the loss-side
  building block for missing-value / imputation training.
- **Needs:** a `task.mode = "imputation"` data path (random input masking, no
  future horizon), an imputation evaluation path, and per-mask-ratio reporting.
- **Why deferred:** changes the task semantics and the four-item dataset contract.

## Other out-of-scope tasks (not planned)

Consistent with the model-porting scope, these remain out of scope (they are
different tasks, not forecasting): **anomaly detection** (PSM/MSL/SMAP/SMD/SWaT),
**classification** (UEA), and **foundation-model pretraining** (e.g. a BLAST-style
corpus, zero-shot LLM forecasters needing billion-scale checkpoints).

## Adopted from the benchmark survey (done)

For reference, the following non-model assets from BasicTS / TSLib / TFB were
adopted (see the relevant docs): extra metrics (`corr`/`rse`/`wape`/`smape`,
`mase` opt-in), masked losses, adjacency-normalization utilities + `adj_norm`,
many CSV + traffic datasets, the pluggable training-callback layer
(curriculum / grad-clip / grad-accum / aux-loss), scaler enhancements,
fit/inference timing, aggregation fairness (`--null-threshold`), the
RollingForecast evaluation strategy, and the dataset-characteristics tool.
