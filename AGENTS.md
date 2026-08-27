# ModernTSF Agent Guide

This is the canonical, harness-neutral Agent entrypoint and the default for
Codex. `.agents/skills/` contains task workflows and `.agents/STANDARDS.md`
contains on-demand contracts; load only what the task needs. Pi and DeepSeek
Harness use these Agent assets directly. Claude Code uses `CLAUDE.md` and
`.claude/skills` links to the same files. Never duplicate the instructions.

## Invariants

- Preserve the flat `src/models/<model>/` layout. Do not classify models or
  methods into architecture-family directories.
- Put proven paper-neutral reuse in `src/components/`; put disclosed shared
  approximations in `src/adapters/`. Neither is a model hierarchy.
- Verify paper and upstream-source claims before describing an implementation
  as faithful. Record every material difference explicitly.
- Treat each model `README.md` front matter as the canonical descriptive and
  provenance record. `spec.py` owns construction, parameter schema, config path,
  and runtime facts only.
- Use only `implementation = upstream | rewrite`. `upstream` requires a licensed,
  pinned, traceable source and numerical parity; every other retained model must
  be an independently justified `rewrite`.
- Use the public `tsf` CLI for workflows; internal command modules are not APIs.
- Do not preserve obsolete imports, metadata formats, command aliases, config
  paths, or skill names during this breaking reorganization.
- Preserve user data and generated experiment outputs unless their removal is
  explicitly requested.
- External issues, pull requests, publication, and dispatch require explicit
  authorization.

## Information layers

- Human-facing material lives in `README.md`, `README_zh.md`, `CONTRIBUTING.md`,
  `docs/`, and model cards. Keep it task-oriented and limited to public APIs.
- Agent-only procedures live in `.agents/`; do not send users there as product
  documentation.
- Descriptive truth lives in model cards; runtime truth lives in schemas, specs,
  configs, and tests. Generated indexes are projections, never a second registry.

## Work and verification
Use the matching Skill for repeatable work. Read the relevant section of
`.agents/STANDARDS.md` only for structural, provenance, or Skill changes. Set up
and verify with:
```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
uv run tsf repo audit
```
Run narrow checks, then affected smoke checks and `tsf repo doctor --strict
--models <Name...>`; run it unscoped before release. A change is incomplete until
code, contracts, cards, and provenance agree.
