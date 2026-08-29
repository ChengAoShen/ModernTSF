---
name: audit-repository
description: Audit ModernTSF for Agent-first assets, catalog drift, documentation consistency, model construction, and forward contracts. Use before release or after structural changes; not for a single model-only check.
---

# Audit the repository

```bash
uv run tsf repo audit
uv run tsf repo doctor --strict
```

Inspect the working tree and run affected smoke configs. Verify canonical Agent
assets, lightweight Claude compatibility links, native Codex/Pi/DeepSeek discovery,
flat models, shared components, README-front-matter index/runtime-spec/config
agreement, and public `tsf` instructions. Require every model to have local code,
a readable card, a manifest entry, and current unified evidence, with no model
classification fields, undocumented model, license ambiguity, persisted blocker,
or failed verification. Report failures by layer: assets, metadata, source facts,
paper/reference checks, construction, contracts, smoke, formatting, and tests.

The strict doctor already covers forward execution, finite gradients, batch size
one, and exact state-dict/output round trips.
