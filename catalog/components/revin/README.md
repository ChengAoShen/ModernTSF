---
name: "revin"
kind: "component"
module: "components.revin"
summary: "Reversible instance normalization."
---

# revin

## Purpose

Reversible instance normalization.

Reversible per-instance normalization for batch-time-channel tensors.

Implementation: [`src/components/revin.py`](../../../src/components/revin.py)

## Public API

- `RevIN(num_features: int, eps: float=1e-05, affine: bool=True, subtract_last: bool=False, enabled: bool=True)`
  Normalize one sequence instance and later restore its original scale.

```python
from components.revin import RevIN
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `revin` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `denormalization`, `instance`, `normalization`, `reversible`.

## Current model consumers

- [`amplifier`](../../../src/models/amplifier/README.md)
- [`amrc`](../../../src/models/amrc/README.md)
- [`aurora`](../../../src/models/aurora/README.md)
- [`catboost_ts`](../../../src/models/catboost_ts/README.md)
- [`cmos`](../../../src/models/cmos/README.md)
- [`cora`](../../../src/models/cora/README.md)
- [`crosslinear`](../../../src/models/crosslinear/README.md)
- [`cyclenet`](../../../src/models/cyclenet/README.md)
- [`decision_tree_ts`](../../../src/models/decision_tree_ts/README.md)
- [`distdf`](../../../src/models/distdf/README.md)
- [`dsformer`](../../../src/models/dsformer/README.md)
- [`duet`](../../../src/models/duet/README.md)
- [`dynamic_tmoe`](../../../src/models/dynamic_tmoe/README.md)
- [`extra_trees_ts`](../../../src/models/extra_trees_ts/README.md)
- [`fets`](../../../src/models/fets/README.md)
- [`film`](../../../src/models/film/README.md)
- [`fredformer`](../../../src/models/fredformer/README.md)
- [`ftp`](../../../src/models/ftp/README.md)
- [`glocalib`](../../../src/models/glocalib/README.md)
- [`gradient_boosting_ts`](../../../src/models/gradient_boosting_ts/README.md)
- [`gru_forecaster_ts`](../../../src/models/gru_forecaster_ts/README.md)
- [`gtr`](../../../src/models/gtr/README.md)
- [`hdmixer`](../../../src/models/hdmixer/README.md)
- [`hn_mvts`](../../../src/models/hn_mvts/README.md)
- [`implicitforecaster`](../../../src/models/implicitforecaster/README.md)
- [`interpdn`](../../../src/models/interpdn/README.md)
- [`lightgbm_ts`](../../../src/models/lightgbm_ts/README.md)
- [`lstm_forecaster_ts`](../../../src/models/lstm_forecaster_ts/README.md)
- [`mgsfformer`](../../../src/models/mgsfformer/README.md)
- [`mlp_forecaster_ts`](../../../src/models/mlp_forecaster_ts/README.md)
- [`moderntcn`](../../../src/models/moderntcn/README.md)
- [`mofo`](../../../src/models/mofo/README.md)
- [`mtsmixer`](../../../src/models/mtsmixer/README.md)
- [`occamvts`](../../../src/models/occamvts/README.md)
- [`olinear`](../../../src/models/olinear/README.md)
- [`paifilter`](../../../src/models/paifilter/README.md)
- [`patchtst`](../../../src/models/patchtst/README.md)
- [`pathformer`](../../../src/models/pathformer/README.md)
- [`phaseformer`](../../../src/models/phaseformer/README.md)
- [`phat`](../../../src/models/phat/README.md)
- [`pmdformer`](../../../src/models/pmdformer/README.md)
- [`pws`](../../../src/models/pws/README.md)
- [`random_forest_ts`](../../../src/models/random_forest_ts/README.md)
- [`rlinear`](../../../src/models/rlinear/README.md)
- [`rnn_forecaster_ts`](../../../src/models/rnn_forecaster_ts/README.md)
- [`sempo`](../../../src/models/sempo/README.md)
- [`srsnet`](../../../src/models/srsnet/README.md)
- [`svtime`](../../../src/models/svtime/README.md)
- [`symtime`](../../../src/models/symtime/README.md)
- [`tcn_forecaster_ts`](../../../src/models/tcn_forecaster_ts/README.md)
- [`texfilter`](../../../src/models/texfilter/README.md)
- [`timealign`](../../../src/models/timealign/README.md)
- [`timecap`](../../../src/models/timecap/README.md)
- [`timeemb`](../../../src/models/timeemb/README.md)
- [`timefilter`](../../../src/models/timefilter/README.md)
- [`timekan`](../../../src/models/timekan/README.md)
- [`timemixer`](../../../src/models/timemixer/README.md)
- [`timemosaic`](../../../src/models/timemosaic/README.md)
- [`timeperceiver`](../../../src/models/timeperceiver/README.md)
- [`timexer`](../../../src/models/timexer/README.md)
- [`tsrag`](../../../src/models/tsrag/README.md)
- [`umixer`](../../../src/models/umixer/README.md)
- [`wavenet`](../../../src/models/wavenet/README.md)
- [`wpmixer`](../../../src/models/wpmixer/README.md)
- [`xgboost_ts`](../../../src/models/xgboost_ts/README.md)
- [`xpatch`](../../../src/models/xpatch/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
