---
name: design-experiment
description: Design a reproducible ModernTSF experiment before execution. Use for selecting baselines, datasets, horizons, metrics, seeds, ablations, controls, budget, and stopping criteria; not for launching an already-defined config.
---

# Design an experiment

Design directly in the current Agent using native reasoning, file editing and
presentation tools. No task renderer or design command is required. Use the
library loader/preflight for scientific contracts; CLI inspection is optional.

State the research question and falsifiable comparison first. Resolve the target task mode, datasets and splits, representative horizons, primary and secondary metrics, strong baselines, controlled variables, seeds, resource budget, and failure or stopping criteria. Separate required comparisons from optional scale-up runs.

Encode shared settings through config inheritance and use `[sweep]` only for intended experimental axes. Keep ablations one-factor-at-a-time unless interactions are the question. Match preprocessing, training budget, evaluation strategy, and metric direction across compared models; disclose unavoidable capability differences.

Prefer a short run file that extends `base.toml`, one dataset preset, and one model
preset. Override only the scientific variables under `experiment`, `task`,
`training`, `model.params`, and `evaluation`; the loader rejects unknown structural
keys and each registry entry validates its own params.

Verify the resolved matrix, run count, and parameter variation through the config loader and preflight API; `uv run tsf inspect --config <run.toml>` is one optional inspection adapter. Deliver the config paths plus a compact design table covering hypothesis, control, treatment, datasets, horizons, metrics, seeds, estimated runs, and acceptance criteria. Do not launch costly runs unless requested.

Once execution is authorized, hand the resolved configs and resource budget to
`run-experiment`; do not duplicate execution instructions here.

When the objective is to match a published table rather than test a new question,
use `reproduce-paper-results` for protocol alignment and claim handling.
