# Submitting results to the TSEval leaderboard

ModernTSF produces results; **TSEval** (on Hugging Face) collects them. A
submission is a self-contained **Submission Report** — your result plus the
agent **trajectory** that produced it plus a human-readable report — uploaded as
a PR. This page is the producer-side workflow.

## 1. Run experiments (optionally with trajectory capture)

Capturing a trajectory is recommended: it is the audit evidence a maintainer
reviews. Capture happens at the `tsf` CLI boundary, so it is **agent-agnostic**
(works the same under Claude Code, Codex, OpenCode, or a human).

```bash
# Start a capture session (optional but recommended)
uv run python tool/tsf.py trace start --label "patchtst-etth1-sweep"

# Run your experiment(s) as usual — every tsf command is recorded
uv run python tool/tsf.py run configs/runs/<your_config>.toml

# End the session
uv run python tool/tsf.py trace end          # or: tsf trace status
```

Each run writes, under `work_dirs/<dataset>/<model>/`:

- `records/<run_id>.json` — a schema-validated `RunRecord` (self-describing
  metrics + profile + env + git SHA). This is what `tsf submit` reads.
- `performance.csv` / `profile.csv` — the usual CSV outputs.

## 2. Package and submit

```bash
# Dry build — assembles the bundle locally, no upload
uv run python tool/tsf.py submit --dataset <DATASET> --model <MODEL> --latest

# Open a PR on the HF Submissions dataset
uv run python tool/tsf.py submit --dataset <DATASET> --model <MODEL> --latest --push
```

`--latest` picks the newest run; use `--run-id <id>` to submit a specific one.
The bundle written to `work_dirs/_submissions/<submission_id>/` contains:

| File | What |
|---|---|
| `submission.json` | the `SubmissionReport` (result + dataset spec + file manifest with sha256) |
| `trajectory.jsonl` | the captured experiment process (or a `synthetic` placeholder if none was captured) |
| `report.html` | a human-readable summary (metrics, profile, environment) |

`--push` opens a PR against `Diaugeia/TSEval-Submissions`. Requires a Hugging
Face login with write access to the `Diaugeia` org (`hf auth login`, or set
`HF_TOKEN`).

## 3. Review & merge

A maintainer opens `report.html` and skims `trajectory.jsonl`, then merges the
PR. v1 review is **human** — no automated agent verification — so the trajectory
is stored as evidence and eyeballed.

## Notes

- No live trajectory session? `tsf submit` still works and writes a minimal
  trajectory marked `synthetic: true`. Capturing a real one is preferred.
- The contract (what a valid submission looks like) is the JSON Schema exported
  by `tsf schema-export` from the `tsf_core` package — the single source of truth
  shared with TSEval.
