---
name: "fred_md"
kind: "dataset"
config: "configs/datasets/fred_md.toml"
loader: "custom"
alias: "fred_md"
mode: "time-series"
summary: "Time-series forecasting preset loaded by `custom`."
---

# fred_md

## Overview

Time-series forecasting preset loaded by `custom`. This card describes the repository preset and runtime contract; it
does not add an upstream provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `custom`
- Config: [`configs/datasets/fred_md.toml`](../../../configs/datasets/fred_md.toml)
- Expected root: `./dataset/FRED-MD`
- Data selector/path: `FRED-MD.csv`
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
    0.7,
    0.1,
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

Inspect availability with `tsf dataset inspect --config configs/datasets/fred_md.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Match the preset's `time-series` layout to the model capability and inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
