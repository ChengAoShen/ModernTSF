---
name: audit-model
description: Audit one existing model or method against its ModelSpec, runtime contract, model card, paper, and upstream source. Use for a focused provenance or implementation review; for a multi-model backlog use audit-model-batch.
---

# Audit a model

```bash
uv run tsf model show <Name>
uv run tsf model audit <Name>
```

Read `spec.py`, preset, card, implementation, cited paper, and authoritative source revision. Compare equations, shapes, defaults, objective, preprocessing, initialization, and output semantics. Record source URL, revision, license, deviations, capabilities, and evidence.

Run `uv run tsf repo doctor --forward --models <Name>`. If `model show` reports a
non-null `smoke_config`, also run `uv run tsf smoke --model <Name>`; most catalog
entries do not own a smoke config. Never promote `unverified` based on naming
similarity or a passing runtime check alone; follow the paper-evidence section of
`.agents/STANDARDS.md`.

For a multi-model verification backlog, use `audit-model-batch` so file ownership,
parallel handoffs, and repository-wide validation remain coordinated.
