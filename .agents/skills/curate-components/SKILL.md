---
name: curate-components
description: Identify repeated blocks across existing ModernTSF models, verify semantic and runtime-contract equivalence, and safely extract reusable components. Use for deliberate cross-model consolidation; not for reorganizing models into families or sharing code based only on similar names.
---

# Curate shared components

Read the components section of `.agents/STANDARDS.md`. Preserve the flat model
layout: components are reusable implementation units, never model categories.

## Qualify candidates

Inventory current components before changing code:

```bash
uv run tsf component list
uv run tsf component match <operation-and-contract-terms> --json
uv run tsf component show <candidate>
```

Search model-local implementations for repeated defining operations. For each
candidate, compare equations, tensor axes and shapes, normalization and residual
order, masking, initialization, state, dtype/device behavior, outputs, and error
conditions. Record a decision of `reuse-existing`, `extract-new`,
`keep-model-local`, or `adapter`; lexical or structural similarity is not enough.

## Extract safely

Define the smallest paper-neutral API that preserves every consumer's behavior.
Add or update its `ComponentSpec`, focused unit tests, and explicit model imports.
Migrate consumers in reviewable groups without flag-driven branches that hide
material variants. Keep attribution and explanatory comments when moving code.

Before deleting local copies, compare affected models on deterministic inputs and
verify parameters, buffers, outputs, gradients, and serialization where relevant.
If equivalence cannot be demonstrated, leave the variants local and document the
reason instead of forcing an abstraction.

Finish with affected tests plus:

```bash
uv run tsf repo doctor --strict --models <Name...>
uv run tsf repo audit
```

Success requires a cataloged component, focused tests, unchanged consumer
contracts, and no peer-model implementation imports.
