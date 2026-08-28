---
name: "dominant_periods"
kind: "component"
module: "components.dominant_periods"
summary: "Top-k FFT period selection with per-sample amplitude weights for BLC tensors."
---

# dominant_periods

## Purpose

Top-k FFT period selection with per-sample amplitude weights for BLC tensors.

FFT-based dominant-period discovery shared by multi-period models.

Implementation: [`src/components/dominant_periods.py`](../../../src/components/dominant_periods.py)

## Public API

- `dominant_periods(x: torch.Tensor, k: int=2)`
  Return top-k integer periods and per-sample FFT amplitudes for BLC data.

```python
from components.dominant_periods import dominant_periods
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `dominant_periods` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `amplitude`, `fft`, `frequency`, `period`, `spectrum`.

## Current model consumers

- [`msgnet`](../../../src/models/msgnet/README.md)
- [`timesnet`](../../../src/models/timesnet/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
