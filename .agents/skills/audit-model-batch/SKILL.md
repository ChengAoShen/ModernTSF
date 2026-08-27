---
name: audit-model-batch
description: Coordinate evidence-based audits of multiple existing ModernTSF models with parallel workers and non-overlapping file ownership. Use for clearing a model-verification backlog; for one model use audit-model, and for finding new papers use discover-papers.
---

# Audit a model batch

Treat the batch as coordinated instances of `audit-model`, not as a weaker bulk
check. Read the paper-evidence section of `.agents/STANDARDS.md` before assigning
work.

## Partition

1. Resolve every requested name with `uv run tsf model list` and inspect its
   current record with `uv run tsf model audit <Name>`.
2. Give each worker exclusive ownership of named model packages and their model
   presets. Keep repository-wide files, shared components, adapters, dependency
   metadata, and notices under one coordinator.
3. Record the initial evidence label and missing paper, source, license, runtime,
   or test facts. Run workers concurrently only when their write sets do not
   overlap.

## Worker contract

For every assigned model, inspect the primary paper and authoritative source,
pin the inspected revision, verify its license, and compare defining equations,
shapes, defaults, preprocessing, objective, initialization, and output semantics
to local code. Preserve useful attribution and implementation comments.

Update only supported facts. A worker handoff must contain model name, files
changed, primary URLs, pinned revision, evidence before and after, mapped
operations, deviations, commands run, and unresolved blockers. Passing shape or
smoke checks alone never promotes evidence.

Run `show` and `audit` for each model, then one targeted contract batch:

```bash
uv run tsf model show <Name>
uv run tsf model audit <Name>
uv run tsf repo doctor --forward --models <Name...>
```

Only run `uv run tsf smoke --model <Name>` when that model's `show` output has a
non-null `smoke_config`.

## Merge and finish

Review handoffs before applying shared-file changes serially. Reject edits that
cross ownership boundaries or make claims beyond recorded evidence. Then run
`uv run tsf repo audit` and `uv run tsf repo doctor --forward`; add `--backward`
when training code or differentiable layers changed.

Continue with disjoint batches until every requested model has either verified
evidence or an explicit blocker. `unverified` with a precise blocker is a valid
audit result, not permission to invent provenance or silently substitute a
different implementation.
