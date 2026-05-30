# Environment setup (hardware-aware)

ModernTSF picks the right **PyTorch build** (CPU or a specific CUDA `cuXXX`) at
install time instead of hard-pinning it. `pyproject.toml` declares plain
`torch==2.6.0` / `torchvision==0.21.0` / `torchaudio==2.6.0` with **no** `+cuXXX`
local version and **no** explicit `[tool.uv.index]` — so the backend is selected
by uv via the `UV_TORCH_BACKEND` environment variable.

---

## How it works

1. **Detect** the GPU and the driver's max CUDA version.
2. **Translate** that into a backend tag (`cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`).
3. **Install** — `uv sync` reads `UV_TORCH_BACKEND` and transparently resolves
   torch from `https://download.pytorch.org/whl/<backend>`.

The agent never downloads wheels by hand: it detects hardware → sets one env var
→ lets `uv sync` fetch the correct build.

---

## Step 1 — Detect hardware

```bash
bash scripts/detect_hardware.sh
```

Example output:

```
gpu=NVIDIA GeForce RTX 4090
driver=550.54.15
cuda=12.4
backend=cu124
```

With no GPU / no `nvidia-smi`, it reports `backend=cpu`. Use `--backend` to print
only the tag (handy for piping into the install command).

## Step 2 — Install with the right backend

Recommended — let uv auto-detect:

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
```

Explicit override (reproducible, or when `auto` guesses wrong):

```bash
UV_TORCH_BACKEND=cu124 uv sync --python 3.12
UV_TORCH_BACKEND=cpu   uv sync --python 3.12
```

Or feed the detector in directly:

```bash
UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync --python 3.12
```

## Step 3 — Verify

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## CUDA ↔ backend ↔ torch table

| Driver CUDA (`nvidia-smi`) | `UV_TORCH_BACKEND` | torch version |
|---|---|---|
| none / no `nvidia-smi` | `cpu` | `2.6.0` (any) |
| 11.8 | `cu118` | `2.6.0` |
| 12.1–12.3 | `cu121` | `2.6.0` |
| 12.4–12.5 | `cu124` | `2.6.0` (project default) |
| 12.6–12.7 | `cu126` | `2.6.0` |
| 12.8+ / 13.x (e.g. Blackwell) | `cu128` | **needs `torch>=2.7`** |

CUDA minor versions are backward compatible within a major release, so a newer
driver can run an older `cuXXX` wheel — pick the highest tag ≤ your driver CUDA.

---

## Different CUDA → different PyTorch version

The project baseline is `torch==2.6.0`, which has wheels for `cpu`/`cu118`/`cu121`/`cu124`/`cu126`.
The `cu128` wheels only exist for `torch>=2.7`, so on a new GPU that requires it,
bump the pins first:

```bash
uv add "torch>=2.7" "torchvision>=0.22" "torchaudio>=2.7"
UV_TORCH_BACKEND=cu128 uv sync --python 3.12
```

---

## Reproducibility & the lock file

- `uv.lock` records the resolved wheels. Switching `UV_TORCH_BACKEND` and running
  `uv sync` **refreshes the lock for that build** — this is expected.
- For a frozen, reproducible environment, commit the `uv.lock` produced **on the
  target hardware** (the lock is backend-specific by nature).
- Do not re-introduce a `+cuXXX` pin or an explicit `[tool.uv.index]` in
  `pyproject.toml`; either would override `UV_TORCH_BACKEND`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` on a GPU box | Re-run `uv sync` with the correct `UV_TORCH_BACKEND` (not `cpu`); confirm `nvidia-smi` works. |
| Wheel error: "no wheel for cu128 / torch 2.6.0" | cu128 needs `torch>=2.7` — bump pins (see above). |
| `auto` resolves to CPU on a GPU machine | `nvidia-smi` not on `PATH`; set the backend explicitly. |
| Sync keeps pulling `+cu124` | A `+cuXXX` pin or `[tool.uv.index]` is still in `pyproject.toml` — remove it. |

See the `setup-env` skill for the agent-facing quick path, and
[scripts.md](scripts.md) for `detect_hardware.sh`.
