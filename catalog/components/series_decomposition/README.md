---
name: "series_decomposition"
kind: "component"
module: "components.series_decomposition"
summary: "Edge-padded moving average and residual/trend decomposition for BLC data."
---

# series_decomposition

## Purpose

Edge-padded moving average and residual/trend decomposition for BLC data.

Edge-padded moving-average decomposition for ``(batch, time, channel)`` data.

Implementation: [`src/components/series_decomposition.py`](../../../src/components/series_decomposition.py)

## Public API

- `EdgePaddedMovingAverage(kernel_size: int, stride: int=1)`
  Smooth a series after repeating its first and last observations.
- `SeriesDecomposition(kernel_size: int)`
  Split ``(B, L, C)`` values into residual and moving-average trend.

```python
from components.series_decomposition import EdgePaddedMovingAverage, SeriesDecomposition
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `series_decomposition` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `decomposition`, `moving-average`, `residual`, `smoothing`, `trend`.

## Current model consumers

- [`amplifier`](../../../src/models/amplifier/README.md)
- [`autoformer`](../../../src/models/autoformer/README.md)
- [`bist`](../../../src/models/bist/README.md)
- [`fedformer`](../../../src/models/fedformer/README.md)
- [`micn`](../../../src/models/micn/README.md)
- [`moderntcn`](../../../src/models/moderntcn/README.md)
- [`stop`](../../../src/models/stop/README.md)
- [`symtime`](../../../src/models/symtime/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
