# Contributing to ModernTSF

Thanks for helping grow the benchmark! This guide covers the common contributions:
adding a model, adding a dataset, and reporting issues.

## Setup

```bash
# The PyTorch build (CPU vs CUDA) is chosen at install time via UV_TORCH_BACKEND.
# Let uv auto-detect, or pin explicitly (cpu / cu121 / cu124 / ...).
UV_TORCH_BACKEND=auto uv sync --python 3.12
bash scripts/detect_hardware.sh   # reports the recommended backend
```

Do **not** add a hardcoded `+cuXXX` torch pin to `pyproject.toml` — it breaks
CPU/macOS installs. The backend is selected via `UV_TORCH_BACKEND`.

## Reporting issues

Open an issue from the templates — **Submit a new model**, **Report a bug**, or
**Ask for a feature**. The forms require the context we need (repro config,
environment, upstream license, …); issues without it may be closed.

## Adding a model

See [`docs/en/add-model.md`](docs/en/add-model.md) (or the `add-model` skill). In short:

1. `src/models/<name>/` — `model.py` (a `Model(nn.Module)` whose
   `forward(self, x, *args)` returns `(B, pred_len, c_out)`), `schema.py`
   (Pydantic `ModelParameterConfig`, `enc_in` required), `registry.py`
   (`register()` calling `MODEL_REGISTRY.register(...)` with a
   `lambda cfg, params: Model(...)` factory), and `__init__.py`.
2. Add the name → module mapping in `src/benchmark/registry/models.py`
   (`MODEL_NAME_MAP`).
3. Add `configs/models/<Name>.toml` and a `configs/runs/smoke_<name>.toml`.
4. Vendor upstream code as `_upstream.py` with a source-URL docstring, and add
   its license to [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
5. Update `docs/en/models.md` + `docs/zh-CN/models.md`.

## Adding a dataset

See [`docs/en/add-dataset.md`](docs/en/add-dataset.md) (or the `add-dataset` skill).

## Verifying

Every model/dataset needs a smoke run that trains 1 epoch and prints Test metrics:

```bash
UV_TORCH_BACKEND=cpu uv run modern-tsf --config configs/runs/smoke_<name>.toml
```

`python scripts/make_smoke_data.py` generates the tiny synthetic datasets the
smoke configs use. For CUDA-kernel-only models that can't run on CPU, document a
forward/shape check instead and note "GPU-untested" in the model docs.

## Licensing

The project is MIT (see [`LICENSE`](LICENSE)). Vendored third-party model code
remains under its **own** upstream license — record it in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and keep the source-URL
docstring in the vendored file.
