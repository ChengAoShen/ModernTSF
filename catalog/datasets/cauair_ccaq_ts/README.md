---
name: "cauair_ccaq_ts"
kind: "dataset"
config: "configs/datasets/cauair_ccaq_ts.toml"
loader: "cauair_ts"
alias: "cauair_ccaq_ts"
mode: "time-series"
summary: "CauAir-style node data flattened to a multivariate time-series preset."
---

# cauair_ccaq_ts

## Overview

CauAir-style node data flattened to a multivariate time-series preset. This card describes the repository preset and runtime contract; it
does not add an upstream provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `cauair_ts`
- Config: [`configs/datasets/cauair_ccaq_ts.toml`](../../../configs/datasets/cauair_ccaq_ts.toml)
- Expected root: `./dataset/cauair_ccaq`
- Data selector/path: `(loader-defined)`
- Track: `standard`

## Input and output contract

Each item contains history/future values shaped `[time, nodes]` plus six-column zero timestamp marks before batching.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "input_dim": 8,
  "npz_name": "his.npz",
  "scale": true
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/cauair_ccaq_ts.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Match the preset's `time-series` layout to the model capability and inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
