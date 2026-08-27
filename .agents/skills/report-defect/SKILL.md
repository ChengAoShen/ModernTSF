---
name: report-defect
description: Reproduce, diagnose, and draft a ModernTSF defect report. Use for bugs, regressions, incorrect model behavior, or repository failures; do not publish an issue or pull request without explicit authorization.
---

# Report a defect

Capture environment, exact public command, minimal config, expected and observed behavior, traceback, and artifacts. Reproduce narrowly, then run relevant checks such as:

```bash
uv run tsf model show <Name>
uv run tsf smoke --model <Name>
uv run tsf repo audit
```

Classify the failure as environment, configuration, dataset, catalog/spec, construction, forward, training, evaluation, or documentation. Determine the cause when possible and draft concise reproduction evidence. Do not implement a fix for a diagnosis-only request or publish externally without authorization.
