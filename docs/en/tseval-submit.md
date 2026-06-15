# Submitting results to the TSEval leaderboard

ModernTSF is where experiments run; [**TSEval**](https://tseval.diaugeia.ai) is
where they are shown, in the open. A submission is not a number you type into a
table — it is a bundle of three things: your result, the agent **trajectory**
that produced it, and a human-readable report. Every row on the board is a
committed submission anyone can open, audit, and reproduce; the board is a
function of the evidence, rebuilt on every submit.

The single source of truth is the leaderboard repository on GitHub:
[github.com/Diaugeia/TSEval](https://github.com/Diaugeia/TSEval). You contribute
a submission by opening a pull request against it — no special access, no manual
table edit. This page is the producer-side workflow.

## 1. Run experiments (optionally with trajectory capture)

Capturing a trajectory is recommended: it is the audit evidence a reviewer
reads. Capture happens at the `tsf` CLI boundary, so it is **agent-agnostic**
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

- `records/<run_id>.json` — a schema-valid `RunRecord` (self-describing
  metrics + profile + env + git SHA). This is what `tsf submit` reads.
- `performance.csv` / `profile.csv` — the usual CSV outputs.

## 2. Build the bundle locally

```bash
# Assemble the submission bundle locally — no upload, no push
uv run python tool/tsf.py submit --dataset <DATASET> --model <MODEL> --latest
```

`--latest` picks the newest run; use `--run-id <id>` to submit a specific one.
The bundle written to `work_dirs/_submissions/<submission_id>/` contains exactly
three files:

| File | What |
|---|---|
| `submission.json` | the `SubmissionReport` (result + dataset spec + file manifest with sha256) |
| `trajectory.jsonl` | the captured experiment process (or a `synthetic` placeholder if none was captured) |
| `report.md` | a human-readable summary (metrics, profile, environment) |

There is **no weight reference** in a submission bundle. A row earns its place
with its result and its process, not its checkpoint — getting onto the board
never requires uploading a `.pth`. If you *want* bit-level reproducibility, you
can optionally archive your trained weights in the public
[TSEval-Weights](https://huggingface.co/datasets/Diaugeia/TSEval-Weights)
dataset, but it is an invitation, never a gate.

## 3. Contribute via a GitHub pull request

The bundle is contributed by adding it to a clone of the leaderboard repo and
opening a PR — there is no `--push`, and nothing is uploaded to Hugging Face.

```bash
# Clone github.com/Diaugeia/TSEval, then copy your bundle dir into place:
#   submissions/<track>/<dataset>/<model>/<run_id>/
#     ├── submission.json
#     ├── trajectory.jsonl
#     └── report.md
# Commit and open a pull request.
```

CI validates your bundle against the TSF-Core JSON Schema, aggregates across
seeds (mean + `n_runs` + per-metric std), and — if it passes — redeploys the
board with your row and its full evidence attached.

## 4. Review & merge

A reviewer opens `report.md` and skims `trajectory.jsonl`, then merges the PR.
v1 review is **human** — no automated agent verification — so the trajectory is
stored as evidence and eyeballed.

## Notes

- No live trajectory session? `tsf submit` still works and writes a minimal
  trajectory marked `synthetic: true`. Capturing a real one is preferred.
- The contract (what a valid submission looks like) is the JSON Schema exported
  by `tsf schema-export` from the `tsf_core` package — the single source of
  truth shared with TSEval, with zero Python coupling (no torch, no ModernTSF
  import on the consumer side).
