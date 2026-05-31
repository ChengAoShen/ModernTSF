---
name: setup-env
description: Detect the machine's GPU and CUDA version and install or repair the project environment with uv, selecting the matching PyTorch backend (CPU or CUDA cuXXX). Use when setting up the project on a new machine, when torch.cuda.is_available() is False or imports fail, when switching hardware, or when the user asks how to install dependencies for their GPU.
---

## When to use

Invoke when the user:
- sets up ModernTSF on a **new machine** and needs the right PyTorch build,
- hits `torch.cuda.is_available() == False`, a CUDA/driver mismatch, or a torch import/wheel error,
- **switches hardware** (CPU ↔ GPU, or a different CUDA version / new GPU like Blackwell),
- asks "how do I install deps for my GPU / what torch version do I need".

The project pins `torch==2.6.0` (and matching `torchvision`/`torchaudio`) **without** a `+cuXXX` tag or explicit index, so the build is chosen at install time by uv's `UV_TORCH_BACKEND`.

## Step 1 — Detect hardware

```bash
bash scripts/detect_hardware.sh
```

Reports `gpu`, `driver`, `cuda`, and a recommended `backend` tag (`cpu | cu118 | cu121 | cu124 | cu126 | cu128`). No GPU / no `nvidia-smi` → `cpu`.

## Step 2 — Install with the right backend

**Recommended — let uv auto-detect the driver and pick the wheel:**

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
```

**Explicit override** (reproducible, or when `auto` guesses wrong):

```bash
UV_TORCH_BACKEND=cu124 uv sync --python 3.12   # pick from the table below
UV_TORCH_BACKEND=cpu   uv sync --python 3.12   # CPU-only / no GPU
```

Or feed the detector straight in:

```bash
UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync --python 3.12
```

## Step 3 — Verify

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## CUDA ↔ backend ↔ torch table

| Driver CUDA (`nvidia-smi`) | `UV_TORCH_BACKEND` | torch version |
|---|---|---|
| none / no `nvidia-smi` | `cpu` | `2.6.0` (any) |
| 11.8 | `cu118` | `2.6.0` |
| 12.1–12.3 | `cu121` | `2.6.0` |
| 12.4–12.5 | `cu124` | `2.6.0` (project default) |
| 12.6–12.7 | `cu126` | `2.6.0` |
| 12.8+ / 13.x (e.g. Blackwell) | `cu128` | **needs `torch>=2.7`** |

## Notes

- **`uv sync` honours `UV_TORCH_BACKEND`** and silently selects the matching `https://download.pytorch.org/whl/<backend>` index — verified on uv 0.9.x. (Do **not** re-add a `+cuXXX` pin or explicit index to `pyproject.toml`; it overrides the backend.)
- **New GPUs needing cu128**: `cu128` wheels only exist for `torch>=2.7`, so bump the pins first, then sync:
  ```bash
  uv add "torch>=2.7" "torchvision>=0.22" "torchaudio>=2.7"
  UV_TORCH_BACKEND=cu128 uv sync --python 3.12
  ```
- The first `uv sync` after switching backend refreshes `uv.lock` for that build — expected. For a frozen per-machine lock, commit the lock produced on that machine.
- Multi-GPU / which GPU to train on is a **runtime** concern, not install: set `experiment.runtime.gpus`/`use_multi_gpu` in the run config, or `--gpus` for `uv run python tool/tsf.py run`.

## Reference

See `docs/en/setup-env.md` for the full mechanism, troubleshooting, and reproducibility notes.
