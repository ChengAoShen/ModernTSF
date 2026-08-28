---
name: add-model
description: Scaffold and integrate a locally implemented forecasting model after its paper structure and runtime contract are resolved. Use for the flat package, model card, spec, config, manifest, and tests; not for paper discovery or placeholder catalog entries.
---

# Add a model

Use the flat `src/models/<module>/` layout. Models and methods are peers; do not create family directories.

1. Resolve the public name, lowercase module slug, parameters, input needs, output
   type, paper identity, and official-code facts when available.
2. Scaffold with `uv run tsf model add --name MyModel --params "enc_in:int,hidden:int=128"`. Add `--graph` only when dataset adjacency is consumed.
3. Before implementing any block, run `uv run tsf component match <requirements> --json` for each operation in the paper structure map and inspect promising candidates with `component show`. Record each operation as `reuse-existing`, `extract-new`, or `model-local`. When an existing component is mathematically and contractually equivalent, reuse it instead of creating a local copy. Retrieval is only a shortlist: verify shapes, axes, normalization, masking, residual order, initialization, state, and outputs first. Replace the placeholder in `model.py`; never import another named model package as an implementation dependency.
4. Complete `spec.py` with only the factory, parameter schema, config path, and
   runtime contract. Put descriptive and provenance facts in README front matter.
5. Complete the model card's method, structure, inputs/outputs, paper and codebase
   links, local implementation, differences, shared components, and constraints.
6. Add the model to `verification/models.toml`, then run `uv run tsf verify model
   MyModel`, `uv run tsf model audit MyModel`, and `uv run tsf repo doctor --strict
   --models MyModel`. Finish with component and repository audits.

Success requires one indexed model card and runtime spec, a preset, passing
unified verification evidence, truthful source facts, and a component decision
for every defining operation. Declare every reused component in `spec.py` and list
it in the model card. Use `implement-model` for the implementation before accepting
the scaffold; generated placeholder code is never a catalog entry.

Use `curate-components` when the task is broader consolidation across existing
models; do not expand a single-model addition into an unsolicited refactor.
