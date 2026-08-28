---
name: audit-model
description: Audit one existing ModernTSF model against its model card, runtime contract, paper, source route, and tests. Use for a focused implementation or provenance review; batch ownership belongs in a task harness.
---

# Audit a model

```bash
uv run tsf model show <Name>
uv run tsf model audit <Name>
```

Treat README front matter as canonical descriptive metadata. Check the card,
`spec.py`, preset, implementation, cited paper, codebase, license, and revision.
Compare equations, shapes, defaults, objective, preprocessing, initialization, and
output semantics. Confirm exactly one route: licensed pinned direct source plus
completed parity for `upstream`, or independent implementation plus paper-structure
validation and no copied unlicensed code for `rewrite`.

Run `uv run tsf repo doctor --strict --models <Name>`. If `model show` reports a
non-null `smoke_config`, also run `uv run tsf smoke --model <Name>`. Check finite
outputs, active gradients, state-dict round trip, CPU, batch and sequence bounds,
and declared marks or adjacency inputs. A passing shape check proves neither
implementation route; report failures instead of persisting blockers in metadata.

When a task assigns several models, apply this complete audit independently to each
named model. Let the task own partitioning, parallel handoffs, shared-file writes,
and the final repository-wide gate.
