---
name: rewrite-model-clean-room
description: Independently implement or replace a ModernTSF forecasting model from papers, textbooks, or method descriptions without copying unlicensed source. Use for every rewrite route, especially when code is absent or reference-only; not for licensed direct ports.
---

# Rewrite a model clean-room

Require a completed paper-structure map. Record any unlicensed repository only as
`codebase.usage: reference-only`; do not inspect or copy its implementation while
writing the replacement. Preserve paper citations, not source-derived comments.

1. Define a local module design from equations, tensor contracts, and disclosed
   inferences. Query the component catalog for every defining operation. Reuse
   an existing component whenever equivalence is proven; otherwise record why a
   new shared extraction or model-local implementation is required.
2. Implement inside the flat model package. Do not import another named model or
   reproduce distinctive unlicensed source structure, naming, or comments.
3. Set README front matter to `implementation: rewrite`. State the basis of the
   implementation, that reference-only source code was not copied, and every
   material paper difference or unresolved detail.
4. Keep `spec.py` limited to construction, parameter schema, config path, and
   runtime facts.
5. Add focused equation/structure tests plus construction, forward/backward,
   finite output, active-gradient, state-dict round-trip, CPU, batch-size,
   sequence-boundary, and declared marks/adjacency checks.

```bash
uv run tsf model show <Name>
uv run tsf model audit <Name>
uv run tsf repo doctor --strict --models <Name>
uv run tsf repo audit
```

Success means the implementation is independently defensible and all claims are
traceable to public method material. Renaming or lightly rearranging an
unlicensed derivative is not a rewrite; replace it completely or report failure.
