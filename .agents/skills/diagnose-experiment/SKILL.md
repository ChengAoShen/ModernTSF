---
name: diagnose-experiment
description: Diagnose a failed, unstable, or invalid ModernTSF experiment. Use for crashes, NaNs, OOMs, suspicious metrics, missing outputs, leakage, or irreproducible runs; not for ordinary result ranking.
---

# Diagnose an experiment

Preserve the failing config, command, environment, logs, and artifacts before changing anything. Reproduce with the smallest equivalent config and classify the failure as environment, data, shape/contract, model, optimization, resource, evaluation, or output bookkeeping.

When the run belongs to a research round, read its complete log and append the
supported failure classification or decision there. Do not paste the full log
into the event stream or create a separate diagnostic ledger.

Inspect the resolved matrix with `uv run tsf inspect`, compare the model's smoke case, and verify dataset splits, tensor shapes, loss/output pairing, metric direction, seeds, device placement, checkpoints, and finite values. For OOM or instability, change one resource or optimization variable at a time; do not silently lower the scientific workload and call it equivalent.

Report the earliest supported root cause, minimal reproduction, evidence, affected runs, and whether existing results are invalid. Apply a fix only when requested, then rerun the minimal reproduction and the affected contract or smoke check. Never overwrite costly artifacts or restart a broad sweep without authorization.

Return repaired runs to `run-experiment` only for the affected scope. Send valid,
comparable completed outputs to `analyze-results`.
