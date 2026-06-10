---
name: report-issue
description: Diagnose and report a ModernTSF framework defect upstream, with user approval, as a GitHub issue or a small verified PR against Diaugeia/ModernTSF. Use after reproducing a crash in src/ or tool/, wrong output or shapes, a broken config or registry, a doc/CLI mismatch, or another repository defect rather than a problem in the user's data or code.
---

## Guardrails

Confirm the defect belongs to ModernTSF, minimize the reproduction, and preserve the user's current work.

- Never publish, push, fork, or open an issue/PR without explicit user approval.
- Show the proposed title and complete body before asking for approval.
- Do not include secrets, private paths, credentials, proprietary data, or unnecessary logs.
- Report one defect at a time. Search open and closed issues/PRs first.
- Do not stage, commit, stash, reset, or otherwise alter unrelated user changes.

## Diagnose and choose

Re-run the smallest failing command and compare expected with actual behavior. Prefer:

- **Issue** when the cause is unclear, the fix needs design input, or a verified fix is not available.
- **PR** for a small, understood fix that can be tested locally.

Before drafting, record:

- exact command and minimal config or input;
- expected and actual behavior;
- complete relevant traceback or logs;
- `bash scripts/detect_hardware.sh`;
- OS, Python, torch, uv, `UV_TORCH_BACKEND`, and `git rev-parse --short HEAD`;
- whether the defect reproduces on current `origin/main`.

Run these preflight checks without changing repository state:

```bash
git status --short --branch
git remote -v
gh auth status
gh search issues "<distinctive error or symptom>" --repo Diaugeia/ModernTSF --state open
gh search issues "<distinctive error or symptom>" --repo Diaugeia/ModernTSF --state closed
gh search prs "<distinctive error or symptom>" --repo Diaugeia/ModernTSF --state open
```

If authentication, network access, or latest-`main` reproduction is unavailable, disclose that in the draft rather than claiming it was checked.

## File an issue

Follow `.github/ISSUE_TEMPLATE/bug_report.yml`, including its duplicate-search and latest-`main` checklist. Use the full traceback requested by the template; redact only sensitive or irrelevant material.

Draft the body in a temporary file so shell quoting cannot corrupt Markdown:

````bash
cat > /tmp/moderntsf-issue.md <<'EOF'
## What happened?
<actual behavior and expected behavior>

## Config to reproduce
```toml
<minimal config>
```

## Command
```shell
<exact command>
```

## Full traceback / logs
```text
<complete relevant output>
```

## Environment
```text
<hardware report, OS, Python, torch, uv, UV_TORCH_BACKEND>
```

## Git commit / version
`<commit>`

## Checklist
- [x] I searched existing issues for a duplicate.
- [x] I reproduced this on the latest `main`.
EOF
````

Leave a checklist item unchecked and explain why if it was not verified. After the user approves the exact title and body:

```bash
gh issue create --repo Diaugeia/ModernTSF \
  --title "[Bug] <short symptom>" \
  --label bug \
  --body-file /tmp/moderntsf-issue.md
```

## Open a PR

Use a separate worktree when the current worktree is dirty or contains unrelated work. Base it on the commit used for verification, preferably a freshly fetched `origin/main`:

```bash
git fetch origin main
git worktree add /tmp/moderntsf-fix-<slug> -b fix/<slug> origin/main
```

Make only the defect fix in that worktree. Verify with the smallest reproduction, then the affected smoke test or another focused check. Review `git diff --check`, `git status --short`, and the full diff before committing.

Follow `.github/PULL_REQUEST_TEMPLATE.md` in the PR body: summary, type of change, exact test commands/results, and applicable checklist items. Clearly mark non-applicable or unverified items. Show the user the final diff summary, title, body, branch, and push destination before asking for approval.

After approval:

```bash
git commit -m "fix(<area>): <symptom>"
git push -u <fork-or-writeable-remote> fix/<slug>
gh pr create --repo Diaugeia/ModernTSF \
  --base main \
  --head <github-user>:fix/<slug> \
  --title "fix(<area>): <symptom>" \
  --body-file /tmp/moderntsf-pr.md
```

If the user lacks upstream write access, create or reuse their fork only after approval. Prefer a named fork remote such as `fork`; do not silently rename or replace `origin`.

## Finish

Return the created URL and resume the original task. If publication fails, report the exact failed step and preserve the draft, branch, and worktree for recovery.
