---
name: "ettm2"
kind: "dataset"
config: "configs/datasets/ettm2.toml"
loader: "ETTm2"
alias: "ettm2"
mode: "time-series"
summary: "Time-series forecasting preset loaded by `ETTm2`."
---

# ettm2

## Overview

Time-series forecasting preset loaded by `ETTm2`. This card describes the repository preset and runtime contract; it
does not add an upstream provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `ETTm2`
- Config: [`configs/datasets/ettm2.toml`](../../../configs/datasets/ettm2.toml)
- Expected root: `./dataset/ETT-small`
- Data selector/path: `ETTm2.csv`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/ettm2.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Match the preset's `time-series` layout to the model capability and inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
