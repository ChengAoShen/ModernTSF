---
name: "pre_processed"
kind: "dataset"
config: "configs/datasets/pre_processed.toml"
loader: "pre_processed"
alias: "pre_processed"
mode: "time-series"
summary: "Time-series forecasting preset loaded by `pre_processed`."
---

# pre_processed

## Overview

Time-series forecasting preset loaded by `pre_processed`. This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `pre_processed`
- Config: [`configs/datasets/pre_processed.toml`](../../../configs/datasets/pre_processed.toml)
- Expected root: `./dataset/my_dataset_npy`
- Data selector/path: `(loader-defined)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/pre_processed.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Match the preset's `time-series` layout to the model capability and inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
