---
name: "marks"
kind: "component"
module: "components.marks"
summary: "Canonical temporal-mark and spatiotemporal input adapters."
---

# marks

## Purpose

Canonical temporal-mark and spatiotemporal input adapters.

Shared input-adaptation helpers for ported external models.

Implementation: [`src/components/marks.py`](../../../src/components/marks.py)

## Public API

- Import the module and use its documented functions/classes.

```python
import components.marks
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `marks` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `calendar`, `covariate`, `spatiotemporal`, `timestamp`.

## Current model consumers

- [`agcrn`](../../../src/models/agcrn/README.md)
- [`aircade`](../../../src/models/aircade/README.md)
- [`airdualode`](../../../src/models/airdualode/README.md)
- [`airformer`](../../../src/models/airformer/README.md)
- [`airphynet`](../../../src/models/airphynet/README.md)
- [`astgcn`](../../../src/models/astgcn/README.md)
- [`bigst`](../../../src/models/bigst/README.md)
- [`bist`](../../../src/models/bist/README.md)
- [`cauair`](../../../src/models/cauair/README.md)
- [`d2stgnn`](../../../src/models/d2stgnn/README.md)
- [`dcrnn`](../../../src/models/dcrnn/README.md)
- [`deepair`](../../../src/models/deepair/README.md)
- [`dfdgcn`](../../../src/models/dfdgcn/README.md)
- [`dgcrn`](../../../src/models/dgcrn/README.md)
- [`gagnn`](../../../src/models/gagnn/README.md)
- [`gclstm`](../../../src/models/gclstm/README.md)
- [`gts`](../../../src/models/gts/README.md)
- [`gwnet`](../../../src/models/gwnet/README.md)
- [`himnet`](../../../src/models/himnet/README.md)
- [`lstm`](../../../src/models/lstm/README.md)
- [`mage`](../../../src/models/mage/README.md)
- [`megacrn`](../../../src/models/megacrn/README.md)
- [`mofo`](../../../src/models/mofo/README.md)
- [`mtgnn`](../../../src/models/mtgnn/README.md)
- [`pcdcnet`](../../../src/models/pcdcnet/README.md)
- [`pm25gnn`](../../../src/models/pm25gnn/README.md)
- [`staeformer`](../../../src/models/staeformer/README.md)
- [`stdn`](../../../src/models/stdn/README.md)
- [`stemgnn`](../../../src/models/stemgnn/README.md)
- [`stgcn`](../../../src/models/stgcn/README.md)
- [`stgode`](../../../src/models/stgode/README.md)
- [`stid`](../../../src/models/stid/README.md)
- [`stnorm`](../../../src/models/stnorm/README.md)
- [`stop`](../../../src/models/stop/README.md)
- [`sttn`](../../../src/models/sttn/README.md)
- [`stwave`](../../../src/models/stwave/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
