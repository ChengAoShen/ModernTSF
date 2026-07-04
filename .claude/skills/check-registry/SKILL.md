---
name: check-registry
description: Check that every configs/models/*.toml has a matching MODEL_NAME_MAP entry (and vice versa) and that every registry entry resolves to a real module. Use after adding/removing/renaming a model, or when a model config or `tsf smoke --model X` says a model can't be found, to find dead configs or dead registry entries.
---

## When to use

The user just added, removed, or renamed a model and wants to confirm the config/registry are still in sync — or a model lookup is failing (`KeyError`, "not registered", `smoke --model X` reports "unknown model") and the cause might be a missing/mismatched entry rather than a code bug.

## Command

```bash
uv run python tool/check_registry.py
```

Static check, no torch import and no model construction — runs in under a second. Exit code `0` and prints `OK: <n> configs/models/*.toml all match MODEL_NAME_MAP 1:1.` when clean. Non-zero exit with a bullet list when it finds:

- a `MODEL_NAME_MAP` entry with no matching `configs/models/<Name>.toml`
- a `configs/models/<Name>.toml` with no matching `MODEL_NAME_MAP` entry
- a `MODEL_NAME_MAP` entry whose module path doesn't resolve to a real file
- two `MODEL_NAME_MAP` entries pointing at the same module

Only covers models — `DATASET_NAME_MAP` is many-to-one (several dataset configs share a `name` like `custom` or `cauair_st`), so a 1:1 check doesn't apply there.

## Reference

Registry pattern (each extensible component has a `*_NAME_MAP` + `register()`): `CLAUDE.md` → "Registry pattern". Adding a model the normal way already keeps these in sync — see the `add-model` skill.
