---
name: extract-paper-structure
description: Extract a forecasting paper's implementable architecture, equations, tensor contracts, official-code clarifications, training objective, and ambiguities. Use before local implementation or structure audit; not for broad literature discovery.
---

# Extract paper structure

Use the primary paper and supplement. Locate official code when available, record
its license, pin a revision, and inspect it to resolve paper omissions. Keep paper
facts, official implementation details, and local design in separate fields; do
not copy source text or code into the implementation map.

Produce a compact implementation map containing:

- task, inputs, covariates, tensor axes, output, and probabilistic semantics;
- preprocessing, normalization, decomposition, embeddings, main blocks, head, and
  inverse transforms in execution order;
- defining equations with paper section or equation references;
- loss, auxiliary objectives, initialization, defaults, and train/eval differences;
- shape invariants, sequence constraints, marks or adjacency contracts, and edge
  cases;
- official-code clarifications with revision and source path when available;
- unspecified details and decisions that would materially change fidelity.

For every implementable operation, include a component decision:
`reuse-existing` with the component name and matched contract, `extract-new`
with expected consumers, or `model-local` with the semantic mismatch. Prefer
`reuse-existing` whenever equivalence is established.

Separate paper facts from implementation inference. Match each operation to an
existing component only after checking mathematics, axes, normalization, masking,
residual order, initialization, state, and outputs with:

```bash
uv run tsf component match <operation-and-contract-terms> --json
uv run tsf component show <candidate>
```

Deliver the map independently of code. It is complete when another worker can
implement or audit the model without guessing a defining operation. Stop and
surface ambiguity when choosing silently would change the named method.
