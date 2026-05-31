---
name: smoke
description: Verify model(s) end-to-end by running their smoke configs concurrently and reporting PASS/FAIL via `tool/tsf.py smoke`. Use when the user wants to check that a model (new or existing) trains and produces the right output shape, validate a port, or run a quick CI-style gate over many models.
---

## When to use

Run this after scaffolding or editing a model, after porting a model, or any time
you want a fast end-to-end check that one or many models register, train for one
epoch on CPU, and emit the correct output shape. Each model has a
`configs/runs/smoke_<module>.toml` (the `new-model` scaffold creates one
automatically).

## Command

```bash
# Verify one model (resolves configs/runs/smoke_<snake(name)>.toml)
uv run python tool/tsf.py smoke --model MyModel

# Verify every model in the repo, 8 at a time
uv run python tool/tsf.py smoke --all --jobs 8

# Verify explicit smoke config(s)
uv run python tool/tsf.py smoke --config configs/runs/smoke_dlinear.toml configs/runs/smoke_gwnet.toml
```

`--jobs N` sets the concurrency (default `min(8, cpu)`). Smoke runs are tiny
(1 epoch, CPU, small data), so high concurrency is safe and fast.

## Output

Per config it prints `PASS` / `FAIL` with the elapsed seconds (and, on failure,
the exit code + last error line), then a final `<passed>/<total> passed` summary.
The command exits non-zero if any config fails — so it works as a CI gate.

## Notes

- A `--model` smoke config must exist; the `new-model` scaffold generates it. If a
  hand-written model lacks one, create `configs/runs/smoke_<module>.toml` (extend
  `../base.toml` + a small dataset + the model config, `epochs = 1`, `device = "cpu"`).
- Graph / spatiotemporal models smoke against `configs/datasets/cauair_ccaq_st.toml`
  with `task.mode = "spatiotemporal"`; plain models against `configs/datasets/smoke.toml`.

## Reference

See `docs/en/scripts.md` for the full `tsf` command reference.
