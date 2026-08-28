---
name: "pems04"
kind: "dataset"
config: "configs/datasets/pems04.toml"
loader: "cauair_st"
alias: "pems04"
mode: "spatiotemporal"
summary: "Node-structured spatiotemporal preset loaded by `cauair_st`."
---

# pems04

## Overview

Node-structured spatiotemporal preset loaded by `cauair_st`. This card describes the repository preset and runtime contract; it
does not add an upstream provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `cauair_st`
- Config: [`configs/datasets/pems04.toml`](../../../configs/datasets/pems04.toml)
- Expected root: `./dataset/pems04`
- Data selector/path: `(loader-defined)`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/pems04.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Match the preset's `spatiotemporal` layout to the model capability and inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
