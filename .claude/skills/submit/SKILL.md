---
name: submit
description: Package a finished run into a TSEval Submission Report (result + agent trajectory + report) and optionally open a Hugging Face PR, via `tsf trace` / `tsf submit`. Use when the user wants to submit results to the TSEval leaderboard, capture an experiment trajectory, or build a leaderboard from submissions.
---

## When to use / what to ask

The user wants to publish a result to the TSEval leaderboard (or audit-package it locally). Ask:

1. **Dataset and model** of the run to submit (must have run already — `tsf submit` reads `work_dirs/<dataset>/<model>/records/<run_id>.json`).
2. **Which run** — `--latest` (newest) or a specific `--run-id`.
3. **Push to Hugging Face?** `--push` opens a PR against `Diaugeia/TSEval-Submissions`; requires HF write access to the `Diaugeia` org (`hf auth login` or `HF_TOKEN`). Without `--push` the bundle is only built locally — confirm before pushing, it publishes the result.

## Capture a trajectory first (recommended)

Start a trace session **before** running the experiment — it records every `tsf` command as audit evidence reviewers read. Agent-agnostic (CLI boundary).

```bash
uv run python tool/tsf.py trace start --label "<experiment-label>"
uv run python tool/tsf.py run configs/runs/<config>.toml   # commands are recorded
uv run python tool/tsf.py trace end                        # or: trace status
```

No session captured? `tsf submit` still works but marks the trajectory `synthetic: true` — a real one is preferred.

## Package and submit

```bash
# Dry build — bundle lands in work_dirs/_submissions/<submission_id>/, no upload
uv run python tool/tsf.py submit --dataset <DATASET> --model <MODEL> --latest

# Same, plus open the HF PR
uv run python tool/tsf.py submit --dataset <DATASET> --model <MODEL> --latest --push
```

The bundle contains `submission.json` (schema-validated `SubmissionReport`), `trajectory.jsonl`, and `report.html`.

## Related commands

```bash
# Collate submissions into a ranked leaderboard.json (consumer side)
uv run python tool/tsf.py leaderboard-build --source work_dirs/_submissions --out leaderboard.json

# Regenerate / verify the JSON Schema contract (CI fails on stale schema)
uv run python tool/tsf.py schema-export [--check]
```

## Reference

Full producer-side workflow and review process: `docs/en/tseval-submit.md`. The contract layer is `src/tsf_core/` (pydantic only).
