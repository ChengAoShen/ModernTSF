---
name: audit-repository
description: Audit ModernTSF for Agent-first assets, catalog drift, documentation consistency, model construction, and forward contracts. Use before release or after structural changes; not for a single model-only check.
---

# Audit the repository

```bash
uv run tsf repo audit
uv run tsf repo doctor --forward
```

Inspect the working tree and run affected smoke configs. Verify canonical Agent
assets, lightweight Claude compatibility links, native Codex/Pi/DeepSeek discovery,
flat models, shared components, README-front-matter index/runtime-spec/config
agreement, and public `tsf` instructions. Require every model to be `upstream` or
`rewrite`, with no undocumented model, license failure, persisted blocker, or
unverified state. Report failures by layer: assets, metadata, licensing, parity or
structure, construction, contracts, smoke, formatting, and tests.

Use `uv run tsf repo doctor --backward` when model training code or shared
differentiable layers changed.
