---
name: extract-paper-structure
description: Extract a forecasting paper's implementable architecture, equations, tensor contracts, training objective, and ambiguities. Use before a clean-room rewrite, upstream comparison, or structure audit; not for broad literature discovery.
---

# Extract paper structure

Use the primary paper and supplement. Treat project pages and repositories as
secondary evidence; do not read unlicensed implementation code when the output will
drive a clean-room rewrite.

Produce a compact implementation map containing:

- task, inputs, covariates, tensor axes, output, and probabilistic semantics;
- preprocessing, normalization, decomposition, embeddings, main blocks, head, and
  inverse transforms in execution order;
- defining equations with paper section or equation references;
- loss, auxiliary objectives, initialization, defaults, and train/eval differences;
- shape invariants, sequence constraints, marks or adjacency contracts, and edge
  cases;
- unspecified details and decisions that would materially change fidelity.

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
