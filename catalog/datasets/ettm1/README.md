---
name: "ettm1"
kind: "dataset"
config: "configs/datasets/ettm1.toml"
loader: "ETTm1"
alias: "ettm1"
task_modes: ["time_series"]
summary: "Time-series forecasting preset loaded by `ETTm1`."
---

# ettm1

## Overview

Time-series forecasting preset loaded by `ETTm1`. This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `ETTm1`
- Config: [`configs/datasets/ettm1.toml`](../../../configs/datasets/ettm1.toml)
- Local path: `./dataset/ETT-small/ETTm1.csv`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "scale": true,
  "split_ratio": [
    0.6,
    0.2,
    0.2
  ],
  "target": "OT"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/ettm1.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Choose one of `time_series` and match it to the model's declared task
mode. Inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
