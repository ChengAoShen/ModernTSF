---
name: "gift_eval/m4_quarterly"
kind: "dataset"
config: "configs/datasets/gift_eval/m4_quarterly.toml"
loader: "gift_eval"
alias: "gift_eval/m4_quarterly"
mode: "gift-eval"
summary: "GIFT-Eval preset for 'm4_quarterly' with forecast horizon 8."
---

# gift_eval/m4_quarterly

## Overview

GIFT-Eval preset for 'm4_quarterly' with forecast horizon 8. This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/m4_quarterly.toml`](../../../../configs/datasets/gift_eval/m4_quarterly.toml)
- Expected root: `./dataset/gift_eval`
- Data selector/path: `m4_quarterly`
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
  "pred_len": 8
}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/gift_eval/m4_quarterly.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Match the preset's `gift-eval` layout to the model capability and inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
