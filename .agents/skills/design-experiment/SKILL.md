---
name: design-experiment
description: Design a reproducible ModernTSF experiment before execution. Use for selecting baselines, datasets, horizons, metrics, seeds, ablations, controls, budget, and stopping criteria; not for launching an already-defined config.
---

# Design an experiment

State the research question and falsifiable comparison first. Resolve the target task mode, datasets and splits, representative horizons, primary and secondary metrics, strong baselines, controlled variables, seeds, resource budget, and failure or stopping criteria. Separate required comparisons from optional scale-up runs.

Encode shared settings through config inheritance and use `[sweep]` only for intended experimental axes. Keep ablations one-factor-at-a-time unless interactions are the question. Match preprocessing, training budget, evaluation strategy, and metric direction across compared models; disclose unavoidable capability differences.

Prefer a short run file that extends `base.toml`, one dataset preset, and one model
preset. Override only the scientific variables under `experiment`, `task`,
`training`, `model.params`, and `evaluation`; the loader rejects unknown structural
keys and each registry entry validates its own params.

Run `uv run tsf inspect --config <run.toml>` to verify the resolved matrix, run count, and parameter variation. Deliver the config paths plus a compact design table covering hypothesis, control, treatment, datasets, horizons, metrics, seeds, estimated runs, and acceptance criteria. Do not launch costly runs unless requested.

Once execution is authorized, hand the resolved configs and resource budget to
`run-experiment`; do not duplicate execution instructions here.

When the objective is to match a published table rather than test a new question,
use `reproduce-paper-results` for protocol alignment and claim handling.
