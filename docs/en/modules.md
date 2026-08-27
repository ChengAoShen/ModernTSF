# Models, components, and adapters

ModernTSF exposes three flat, orthogonal catalogs. They are organization
boundaries, not architecture families:

- **Models and methods** are named runnable entries under `src/models/<slug>/`.
  Each owns a `ModelSpec`, parameter schema, preset, implementation wrapper,
  model card, provenance status, and tensor contract.
- **Components** are reusable mathematical building blocks under
  `src/components/`. A component is extracted only when all consumers share
  the same behavior and shape semantics.
- **Adapters** are disclosed approximation backends under `src/adapters/`.
  Their consumers carry `evidence="adaptation"`; they are not paper
  reproductions.

Use the catalogs without importing implementations:

```bash
uv run tsf model list --details
uv run tsf model show DLinear
uv run tsf component list
uv run tsf component match "patch transformer normalization"
uv run tsf component show quantile_head
uv run tsf adapter list
uv run tsf adapter show recent-tsf
```

`tsf model show` includes the introductory method description from the model
card. Component output includes its semantic contract, public symbols, and
consumer packages. Adapter output includes its limitation and every catalog
entry that uses it.

`component match` ranks retrieval candidates from contract terms; it does not
assert that two implementations are mathematically interchangeable. Inspect a
candidate with `component show` and verify its semantics before reuse.

Resolve runnable entries through the public CLI so construction, parameter
validation, evidence, and capabilities remain attached.
