---
name: "pems_bay"
kind: "dataset"
config: "configs/datasets/pems_bay.toml"
loader: "cauair_st"
alias: "pems_bay"
task_modes: ["spatiotemporal", "covariate"]
summary: "Node-structured spatiotemporal preset loaded by `cauair_st`."
---

# pems_bay

## Overview

Node-structured spatiotemporal preset loaded by `cauair_st`. This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `cauair_st`
- Config: [`configs/datasets/pems_bay.toml`](../../../configs/datasets/pems_bay.toml)
- Local path: `./dataset/pems_bay`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item is `(value_history, value_future, covariate_history, covariate_future)`; values use `[time, nodes]` and covariates `[time, nodes, features]` before batching.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "input_dim": 3,
  "npz_name": "his.npz",
  "scale": false
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/pems_bay.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Choose one of `spatiotemporal, covariate` and match it to the model's declared task
mode. Inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
