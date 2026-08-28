# Contributing to ModernTSF

Thanks for helping grow the benchmark! This guide covers the common contributions:
adding a model, adding a dataset, and reporting issues.

## Branching & releases

- **`dev` is the integration branch — open your PRs against `dev`, not `main`.**
  New models, datasets, bug fixes, and features all land on `dev` first.
- **`main` is release-only and versioned.** It is protected (a PR is required to
  merge, the `schema-check` status check must pass, force-pushes and deletion are
  blocked). `main` normally advances only by promoting `dev` → `main`, and
  **every `main` update bumps the version (`pyproject.toml` + `CHANGELOG.md`) and
  ships a tagged GitHub Release** (`vX.Y.Z`, [semver](https://semver.org/)):
  bug-fix-only promotions bump the patch, new models/features bump the minor.
- A maintainer may push an urgent hot-fix straight to `main` (admin bypass), but
  it still must carry the version bump + release.
- Branch names follow the conventional scopes used in history: `feat/…`,
  `fix/…`, `docs/…`, `chore/…`.

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

See [`docs/en/add-model.md`](docs/en/add-model.md). In short:

1. Run `uv run tsf model add --name MyModel --params "enc_in:int"`.
2. Implement `src/models/<name>/model.py` and complete its peer-level
   `spec.py`. Models and methods use the same flat namespace; do not add family
   or architecture directories.
3. Complete `configs/models/<Name>.toml`,
   `configs/runs/smoke_<name>.toml`, and the model card.
4. Reuse paper-neutral code through `src/models/_components/`; keep paper-specific
   operations model-local unless output and gradient equivalence are proven.
5. Choose one provenance route: a clean-room `rewrite`, or a licensed `upstream`
   port pinned to a revision with executable numerical parity. Record the route
   in the model card and update [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
   only for an actual upstream port. Approximations cannot be registered as a
   completed model.
6. Run the catalog, forward, backward, verification, strict, and smoke checks; generated model tables
   must not be edited by hand.

## Adding a dataset

See [`docs/en/add-dataset.md`](docs/en/add-dataset.md).

## Verifying

Every model/dataset needs a smoke run that trains 1 epoch and prints Test metrics:

```bash
UV_TORCH_BACKEND=cpu uv run tsf smoke --config configs/runs/smoke_<name>.toml
uv run tsf repo doctor --backward
```

`python scripts/make_smoke_data.py` generates the tiny synthetic datasets the
smoke configs use. For CUDA-kernel-only models that can't run on CPU, document a
forward/shape check instead and note "GPU-untested" in the model docs.

## Licensing

The project is MIT (see [`LICENSE`](LICENSE)). Vendored third-party model code
remains under its **own** upstream license — record it in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and keep the source-URL
docstring in the vendored file.
