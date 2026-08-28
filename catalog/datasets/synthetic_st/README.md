---
name: "synthetic_st"
kind: "dataset"
config: "configs/datasets/synthetic_st.toml"
loader: "synthetic_st"
alias: "synthetic_st"
mode: "spatiotemporal"
summary: "Node-structured spatiotemporal preset loaded by `synthetic_st`."
---

# synthetic_st

## Overview

Node-structured spatiotemporal preset loaded by `synthetic_st`. This card describes the repository preset and runtime contract; it
does not add an upstream provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `synthetic_st`
- Config: [`configs/datasets/synthetic_st.toml`](../../../configs/datasets/synthetic_st.toml)
- Expected root: ``
- Data selector/path: `(loader-defined)`
- Track: `standard`

## Input and output contract

Each item is `(value_history, value_future, covariate_history, covariate_future)`; values use `[time, nodes]` and covariates `[time, nodes, features]` before batching.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "length": 600,
  "num_nodes": 8,
  "split_ratio": [
    0.6,
    0.2,
    0.2
  ],
  "steps_per_day": 24
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/synthetic_st.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Match the preset's `spatiotemporal` layout to the model capability and inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
