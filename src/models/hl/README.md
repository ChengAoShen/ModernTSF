---
name: "HL"
implementation: rewrite
summary: "HL (Historical Last) is a naive spatiotemporal forecasting baseline that repeats the last observed value across every node and every step of the prediction horizon. It serves as a lower-bound reference in graph- and node-structured benchmarks, providing the simplest possible prediction without any learning."
paper:
  title: "Historical-last persistence baseline (no associated paper)"
  venue: "N/A (classical baseline)"
  year: null
  url: ""
codebase:
  url: "https://github.com/PoorOtterBob/CauAir"
  revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
  license: "NOASSERTION"
  usage: reference-only
---
# HL

HL (Historical Last) is a naive spatiotemporal forecasting baseline that repeats the last observed value across every node and every step of the prediction horizon. It serves as a lower-bound reference in graph- and node-structured benchmarks, providing the simplest possible prediction without any learning.

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Historical Last (HL) is a naive persistence baseline for spatiotemporal forecasting. For each node in the graph, it copies the final observed value from the input window and repeats it identically across all future time steps in the prediction horizon. Despite its simplicity, the method provides a meaningful lower bound: any learned model should outperform HL, especially over longer horizons where temporal dynamics diverge from the last observation. Its forecast has no effective learned behavior; the implementation retains only a zero-contribution dummy parameter path for the shared optimizer/backward contract.

## In ModernTSF
Default config: `configs/models/HL.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

Implementation: **rewrite** (clean-room audit pending). There is no associated paper or canonical
author implementation. The immediate source is
[`PoorOtterBob/CauAir`](https://github.com/PoorOtterBob/CauAir) at revision
`73dae00ca6ad14abb15174a0a0286d500e868b94`, whose repository declares no
license. The auditable behavior is limited to the persistence formula: repeat
the final observed value for every horizon step.

## Citation

HL is an in-repository persistence baseline and has no associated paper or
canonical BibTeX entry.
