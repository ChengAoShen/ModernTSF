<!-- Thanks for contributing to ModernTSF! Fill in the sections below. -->

## Summary

<!-- What does this PR do and why? -->

## Type of change

- [ ] New model
- [ ] New dataset
- [ ] Bug fix
- [ ] Feature / framework change
- [ ] Docs only

## How was it tested?

<!-- Paste the smoke command(s) and their result. Models/datasets must have a
     smoke config that trains for 1 epoch and reports Test metrics on CPU, e.g.:
       UV_TORCH_BACKEND=cpu uv run modern-tsf --config configs/runs/smoke_<name>.toml
     If the model needs CUDA kernels and can't run on CPU, say so and describe
     the shape/forward check you did instead. -->

## Checklist

- [ ] Smoke run passes (or shape-check documented for CUDA-only models).
- [ ] New model/dataset is registered in the relevant `*_NAME_MAP` and has a config under `configs/`.
- [ ] Vendored upstream code keeps a source-URL docstring; `THIRD_PARTY_NOTICES.md` updated with its license.
- [ ] Docs updated (`docs/en/` and `docs/zh-CN/` mirror) and model/dataset tables refreshed.
- [ ] No hardcoded `+cuXXX` torch pin added to `pyproject.toml` (the build is chosen via `UV_TORCH_BACKEND`).
