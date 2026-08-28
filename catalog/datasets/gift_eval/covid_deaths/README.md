---
name: "gift_eval/covid_deaths"
kind: "dataset"
config: "configs/datasets/gift_eval/covid_deaths.toml"
loader: "gift_eval"
alias: "gift_eval/covid_deaths"
task_modes: ["time_series"]
summary: "GIFT-Eval preset for 'covid_deaths' with forecast horizon 30."
---

# gift_eval/covid_deaths

## Overview

GIFT-Eval preset for 'covid_deaths' with forecast horizon 30. This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/covid_deaths.toml`](../../../../configs/datasets/gift_eval/covid_deaths.toml)
- Expected root: `./dataset/gift_eval`
- Data selector/path: `covid_deaths`
- Track: `standard`

## Input and output contract

Windowed history/target values and timestamp marks; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "scale": true
}
```

## Task overrides

```json
{
  "features": "M",
  "pred_len": 30
}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/gift_eval/covid_deaths.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Choose one of `time_series` and match it to the model's declared task
mode. Inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
