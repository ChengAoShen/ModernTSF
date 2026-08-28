---
name: "graph_spectral"
kind: "component"
module: "components.graph_spectral"
summary: "Robust scaled-Laplacian and exact-order Chebyshev support construction."
---

# graph_spectral

## Purpose

Robust scaled-Laplacian and exact-order Chebyshev support construction.

Robust scaled-Laplacian and Chebyshev supports for spectral graph models.

Implementation: [`src/components/graph_spectral.py`](../../../src/components/graph_spectral.py)

## Public API

- `scaled_laplacian(adj_mx: np.ndarray, *, undirected: bool=True)`
  Return a dense scaled normalized Laplacian for any finite square graph.
- `chebyshev_polynomials(matrix: np.ndarray, order: int)`
  Return exactly ``order`` Chebyshev polynomials, beginning with identity.
- `chebyshev_supports(adj_mx: np.ndarray, order: int, *, undirected: bool=True)`
  Build exactly ``order`` dense Chebyshev supports for an adjacency matrix.

```python
from components.graph_spectral import scaled_laplacian, chebyshev_polynomials, chebyshev_supports
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `graph_spectral` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adjacency`, `chebyshev`, `degenerate`, `graph`, `laplacian`, `spectral`.

## Current model consumers

- [`astgcn`](../../../src/models/astgcn/README.md)
- [`dstagnn`](../../../src/models/dstagnn/README.md)
- [`gclstm`](../../../src/models/gclstm/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
