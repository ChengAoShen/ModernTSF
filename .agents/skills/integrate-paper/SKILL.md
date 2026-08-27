---
name: integrate-paper
description: Route an approved forecasting paper into a verified ModernTSF model entry. Use after discovery to coordinate structure extraction, route selection, component decisions, implementation, and integration; not for literature scanning alone.
---

# Integrate a paper

Require a primary paper URL, deduplication result, and authorization to implement.
Use `extract-paper-structure` to resolve defining operations and runtime inputs.
Choose exactly one route: `port-upstream-model` only for a licensed authoritative
source, otherwise `rewrite-model-clean-room`. An unlicensed source is
`reference-only` and cannot supply implementation code.

Build a reuse decision table before code:

```bash
uv run tsf component match <operation-and-contract-terms> --json
uv run tsf component show <candidate>
uv run tsf adapter list
```

Classify each required block as `reuse`, `extend`, `model-local`, or `adapter`. Lexical matches are candidates only; compare mathematics, shapes, normalization, masking, residual order, initialization, and output semantics. Extend a shared component only when existing consumers retain their contracts.

Scaffold through `add-model`. Put descriptive and provenance facts in README front
matter; keep `spec.py` limited to construction, parameter schema, config path, and
runtime facts. Preserve useful attribution. Finish with focused component tests,
`model show`, model audit, forward/backward contracts, and `repo audit`. For an
upstream route, `verify-upstream-parity` must pass before integration is complete.

Use `curate-components` for consolidation spanning existing models. After the model
is runnable and audited, use `reproduce-paper-results` only when matching the
paper's reported experiments is requested.
