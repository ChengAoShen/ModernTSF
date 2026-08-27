---
name: submit-results
description: Capture an experiment trajectory and package a completed ModernTSF run for TSEval submission. Use for local submission bundles or leaderboard contribution; publishing a pull request requires explicit authorization.
---

# Submit results

```bash
uv run tsf trace start --label <label>
uv run tsf run configs/runs/<run>.toml
uv run tsf trace end
uv run tsf submit --dataset <dataset> --model <model> --latest
uv run tsf schema-export --check
```

Inspect `submission.json`, `trajectory.jsonl`, and `report.md`; confirm dataset version, run identity, metrics, and synthetic-trajectory status. Use `uv run tsf leaderboard-build` for a local board. A remote branch, issue, or pull request requires explicit user approval.
