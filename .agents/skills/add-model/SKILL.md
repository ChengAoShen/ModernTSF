---
name: add-model
description: Scaffold and integrate a new forecasting model or method in ModernTSF. Use when adding a model package, parameter contract, catalog entry, model card, or smoke case; not for auditing an existing implementation.
---

# Add a model

Use the flat `src/models/<module>/` layout. Models and methods are peers; do not create family directories.

1. Resolve the public model name, lowercase module slug, parameters, input needs, and output type.
2. Scaffold with `uv run tsf model add --name MyModel --params "enc_in:int,hidden:int=128"`. Add `--graph` only when dataset adjacency is consumed.
3. Run `uv run tsf component match <requirements> --json`, inspect promising candidates with `component show`, and list adapters before writing code. Retrieval is only a shortlist: reuse a component only after its mathematics, shapes, normalization, masking, residual order, initialization, and outputs match. Replace the placeholder in `model.py`; never import another named model package as an implementation dependency.
4. Complete `spec.py`: factory, parameter schema, capabilities, paper/source metadata, deviations, evidence, and contract task.
5. Replace the scaffolded model-card paragraph with a concise method description, then update the evidence and preset. Do not claim reproduction before verification.
6. Run `uv run tsf model show MyModel`, `uv run tsf smoke --model MyModel`, and `uv run tsf repo doctor --forward`.

Success requires one catalog spec and preset, passing construction/forward/smoke checks, and truthful provenance. Stop if the paper, license, input contract, or output semantics need a user decision.

Use `curate-components` when the task is broader consolidation across existing
models; do not expand a single-model addition into an unsolicited refactor.
