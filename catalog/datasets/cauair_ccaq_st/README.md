---
name: "cauair_ccaq_st"
kind: "dataset"
config: "configs/datasets/cauair_ccaq_st.toml"
loader: "cauair_st"
alias: "cauair_ccaq_st"
task_modes: ["spatiotemporal", "covariate"]
summary: "Node-structured spatiotemporal preset loaded by `cauair_st`."
---

# cauair_ccaq_st

## Overview

Node-structured spatiotemporal preset loaded by `cauair_st`. This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `cauair_st`
- Config: [`configs/datasets/cauair_ccaq_st.toml`](../../../configs/datasets/cauair_ccaq_st.toml)
- Local path: `./dataset/cauair_ccaq`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item is `(value_history, value_future, covariate_history, covariate_future)`; values use `[time, nodes]` and covariates `[time, nodes, features]` before batching.

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

Inspect availability with `tsf dataset inspect --config configs/datasets/cauair_ccaq_st.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Choose one of `spatiotemporal, covariate` and match it to the model's declared task
mode. Inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
