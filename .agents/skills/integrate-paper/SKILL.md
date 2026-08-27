---
name: integrate-paper
description: Turn an approved forecasting paper candidate into a verified ModernTSF model entry. Use when paper discovery is complete and the user wants source review, component reuse decisions, implementation, and integration; not for literature scanning alone.
---

# Integrate a paper

Require a primary paper URL, deduplication result, and explicit authorization to implement. Resolve the authoritative source, pinned revision, license, tensor inputs and outputs, defining operations, defaults, and likely deviations before scaffolding. Stop when identity, licensing, or runtime intent needs a user decision.

Build a reuse decision table before code:

```bash
uv run tsf component match <operation-and-contract-terms> --json
uv run tsf component show <candidate>
uv run tsf adapter list
```

Classify each required block as `reuse`, `extend`, `model-local`, or `adapter`. Lexical matches are candidates only; compare mathematics, shapes, normalization, masking, residual order, initialization, and output semantics. Extend a shared component only when existing consumers retain their contracts.

Scaffold through `uv run tsf model add`, implement the named package, and record paper/source evidence, deviations, capabilities, adapter, and exact component imports in `spec.py`. Preserve useful upstream comments and attribution. Finish with focused component tests, `model show`, model audit, forward/backward contracts, and `repo audit`. Run `tsf smoke --model` only when `model show` reports a non-null `smoke_config`. A passing tensor check does not prove paper fidelity.

Use `curate-components` for consolidation spanning existing models. After the model
is runnable and audited, use `reproduce-paper-results` only when matching the
paper's reported experiments is requested.
