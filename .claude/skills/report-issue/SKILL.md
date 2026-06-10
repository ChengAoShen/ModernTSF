---
name: report-issue
description: Report a ModernTSF framework defect upstream — ask the user whether to file a GitHub issue or open a PR against Diaugeia/ModernTSF. Use when, while running experiments or using the tools, you discover a bug, crash, wrong result, broken config/doc, or other defect in ModernTSF itself (not in the user's own code or data).
---

## When to use

You hit something broken **in the framework** while doing other work: a crash inside `src/` or `tool/`, a model producing wrong shapes, a config that doesn't expand as documented, a doc/CLI mismatch, a stale or missing registry entry. First make sure it is not the user's own config/data/code at fault.

**Never file anything without asking.** Filing an issue or PR publishes to GitHub. Present the problem to the user and ask: report it as an **issue**, fix it and open a **PR**, or **skip**.

## Decide: issue or PR

- **Issue** — cause unclear, fix non-trivial, or touches design decisions.
- **PR** — you have a small, verified fix (typo, off-by-one, missing registry entry, doc fix). Verify first: `uv run python tool/tsf.py smoke --model <affected>` or re-run the failing command.

## File an issue

Gather before filing: the exact command + config, expected vs actual, the traceback (trimmed to the relevant frames), and env (`uv run python -c "import torch,sys;print(sys.version,torch.__version__)"`, platform, `git rev-parse HEAD`).

```bash
gh issue create --repo Diaugeia/ModernTSF \
  --title "<one-line symptom>" \
  --body "$(cat <<'EOF'
## What happened
<expected vs actual>

## Repro
```bash
<exact command>
```
Config: <path or inline snippet>

## Traceback / output
<trimmed>

## Environment
<python / torch / platform / commit SHA>
EOF
)"
```

## Open a PR

Work on a branch, keep the diff minimal (the fix only — no drive-by changes), verify, then:

```bash
git checkout -b fix/<short-slug>
# ...commit the fix (conventional message, e.g. "fix(registry): ...")...
git push -u origin fix/<short-slug>
gh pr create --repo Diaugeia/ModernTSF \
  --title "fix(<area>): <symptom>" \
  --body "<what was broken, how reproduced, what the fix does, how verified>"
```

No write access to the repo? Fork first (`gh repo fork Diaugeia/ModernTSF --remote`), push the branch to the fork, then `gh pr create` the same way.

## Notes

- Show the user the drafted title/body before running `gh` — it's their name on the report.
- One defect per issue/PR; if you found several, list them and let the user pick.
- After filing, return to the original task; mention the issue/PR URL in your summary.
