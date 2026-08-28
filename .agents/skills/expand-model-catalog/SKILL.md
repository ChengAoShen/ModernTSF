---
name: expand-model-catalog
description: Discover and integrate a bounded number of new time-series forecasting papers into the ModernTSF catalog. Use for an authorized search-to-model expansion run; not for literature monitoring that must leave the repository unchanged.
---

# Expand the model catalog

Require authorization to modify the repository and an explicit candidate/model
budget. Use `discover-papers` to search arXiv and Hugging Face Papers, normalize
identities, and deduplicate against `uv run tsf model list --json`. No candidate
clearing the gate is a successful no-change outcome.

For each retained candidate, verify the primary paper, forecasting task, required
inputs, authoritative source, revision, and license before selecting at most the
authorized number. Use `extract-paper-structure`, then `integrate-paper`. Choose
`upstream` only when a licensed pinned port passes numerical parity; otherwise use
a paper-derived clean-room `rewrite`. Never use search relevance or a shape-only
test as implementation evidence.

Inspect `tsf component match` before writing code, but keep material variants
model-local. Preserve the flat model layout, useful paper comments, and every
known difference. Each added model must pass focused tests, strict runtime,
model audit, repository audit, and its route-specific verification evidence.

Do not publish, push, open issues, or dispatch more tasks unless separately
authorized. Stop when the model budget is exhausted or paper, license, data, or
runtime ambiguity would require inventing a claim.
