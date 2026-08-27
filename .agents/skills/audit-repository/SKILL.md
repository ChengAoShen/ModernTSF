---
name: audit-repository
description: Audit ModernTSF for Agent-first assets, catalog drift, documentation consistency, model construction, and forward contracts. Use before release or after structural changes; not for a single model-only check.
---

# Audit the repository

```bash
uv run tsf repo audit
uv run tsf repo doctor --forward
```

Inspect the working tree and run affected smoke configs. Verify canonical Agent assets, Claude Code compatibility links, native Codex/Pi/DeepSeek Harness discovery, flat models, shared components, catalog/spec/preset/card agreement, and public `tsf` instructions. Report failures by layer: assets, catalog, docs, construction, forward, smoke, formatting, and tests. Static success alone is not release readiness.

Use `uv run tsf repo doctor --backward` when model training code or shared
differentiable layers changed.
