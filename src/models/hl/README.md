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

<!-- model-card:canonical:start -->
## Method overview

HL (Historical Last) is a naive spatiotemporal forecasting baseline that repeats the last observed value across every node and every step of the prediction horizon.

## Core architecture

It serves as a lower-bound reference in graph- and node-structured benchmarks, providing the simplest possible prediction without any learning.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Graph adjacency is supplied at construction; temporal/node covariates follow the runtime batch contract.

## Paper and code

- paper: not available; title: Historical-last persistence baseline (no associated paper); venue/year: N/A (classical baseline) / not available
- [codebase](https://github.com/PoorOtterBob/CauAir); revision: `73dae00ca6ad14abb15174a0a0286d500e868b94`; license: `NOASSERTION`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/HL.toml`](../../../configs/models/HL.toml).

## Differences

Implementation: **rewrite** (clean-room audit pending). There is no associated paper or canonical
author implementation. The immediate source is
[`PoorOtterBob/CauAir`](https://github.com/PoorOtterBob/CauAir) at revision
`73dae00ca6ad14abb15174a0a0286d500e868b94`, whose repository declares no
license. The auditable behavior is limited to the persistence formula: repeat
the final observed value for every horizon step.

## Shared components

- [`marks`](../../components/marks.py)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`
<!-- model-card:canonical:end -->

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
