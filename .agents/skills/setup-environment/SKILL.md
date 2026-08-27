---
name: setup-environment
description: Install, repair, or verify the ModernTSF Python environment and PyTorch backend. Use for first-time setup, dependency failures, CUDA detection problems, or hardware changes.
---

# Set up the environment

Run `bash scripts/detect_hardware.sh`, then:

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run tsf --help
uv run tsf repo audit
```

Use an explicit backend only when auto-detection is wrong or reproducibility requires it. Do not change dependency pins to mask a driver mismatch. Report Python and torch versions, selected backend, accelerator visibility, and lockfile changes.
