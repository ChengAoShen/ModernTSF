# ModernTSF Agent Guide

This is the small, always-loaded Agent layer. `.agents/skills/` contains
task-specific workflows and `.agents/STANDARDS.md` contains detailed contracts;
load only what the current task needs. Codex, Pi, and DeepSeek Harness read the
canonical Agent files directly. Claude Code uses `CLAUDE.md` and
`.claude/skills` compatibility links; never author duplicate instructions there.

## Invariants

- Preserve the flat `src/models/<model>/` layout. Do not classify models or
  methods into architecture-family directories.
- Put proven paper-neutral reuse in `src/components/`; put disclosed shared
  approximations in `src/adapters/`. Neither is a model hierarchy.
- Verify paper and upstream-source claims before describing an implementation
  as a reproduction. Record adaptations and unknowns explicitly.
- Treat each model's `spec.py` as the source for identity, construction,
  parameters, provenance, runtime contract, and verification status.
- Use the public `tsf` CLI for workflows; internal command modules are not APIs.
- Do not preserve obsolete imports, registry formats, command aliases, config
  paths, or skill names during the current breaking reorganization.
- Preserve user data and generated experiment outputs unless their removal is
  explicitly requested.
- External issues, pull requests, publication, and dispatch require explicit
  authorization.

## Information layers

- Human-facing material lives in `README.md`, `README_zh.md`, `CONTRIBUTING.md`,
  `docs/`, and model cards. Keep it task-oriented and limited to public APIs.
- Agent-only procedures live in `.agents/`; do not send users there as product
  documentation.
- Executable truth lives in schemas, specs, catalogs, configs, and tests.
  Generated documentation is a projection, never a second registry.

## Work and verification

Use the matching Skill for repeatable work. Read the relevant section of
`.agents/STANDARDS.md` only for structural, evidence, or Skill changes. Set up
and verify with:

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
uv run tsf repo audit
```

Run the narrowest relevant checks while developing, then the repository audit
and affected smoke checks. A change is incomplete until code, executable
contracts, human documentation, and verification evidence agree.
