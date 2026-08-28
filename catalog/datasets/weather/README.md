---
name: "weather"
kind: "dataset"
config: "configs/datasets/weather.toml"
loader: "weather"
alias: "weather"
mode: "time-series"
summary: "Time-series forecasting preset loaded by `weather`."
---

# weather

## Overview

Time-series forecasting preset loaded by `weather`. This card describes the repository preset and runtime contract; it
does not add an upstream provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `weather`
- Config: [`configs/datasets/weather.toml`](../../../configs/datasets/weather.toml)
- Expected root: `./dataset/weather`
- Data selector/path: `weather.csv`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/weather.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Match the preset's `time-series` layout to the model capability and inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
