---
name: check-docs
description: Check that model counts in CLAUDE.md/docs/en/models.md match configs/models/*.toml, that every docs/{en,zh-CN}/*.md page is linked from its README index, and that the en/zh doc trees mirror each other. Use after adding/removing a model, adding a new docs page, or before a release to catch stale counts and orphaned pages.
---

## When to use

Right after adding/removing a model (counts drift), after adding a new `docs/en/*.md` page (easy to forget the README index link and the zh-CN mirror), or as a pre-release sanity pass. This is a text-only static check — it doesn't tell you if the *content* is accurate, only that counts/links/mirrors are self-consistent.

## Command

```bash
uv run python tool/check_docs.py
```

No torch import, runs in under a second. Exit `0` and prints `OK: model counts and docs index/mirror are consistent.` when clean. Non-zero exit with a bullet list when it finds:

- `CLAUDE.md`'s `## Available Models (N)` heading, its category-count table's `Total` row, or `docs/en/models.md`'s `includes N models` sentence disagreeing with the actual `configs/models/*.toml` count (or the category counts not summing to the table's Total)
- a `docs/en/*.md` or `docs/zh-CN/*.md` page not linked from its directory's `README.md` index
- a page that exists in `docs/en/` but not `docs/zh-CN/` (or vice versa)

Doesn't check narrative accuracy (e.g. a CHANGELOG entry missing for a shipped feature) — that class of drift needs a human/review pass, not a regex.

## Reference

Docs are English + Chinese mirror pairs, indexed by `docs/{en,zh-CN}/README.md`: see `CLAUDE.md` → "Detailed docs". For registry (not docs) consistency, see the `check-registry` skill.
