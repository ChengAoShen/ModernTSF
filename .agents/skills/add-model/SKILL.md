---
name: add-model
description: Scaffold and integrate a forecasting model after its implementation route is resolved. Use for adding the flat model package, model card, runtime spec, config, and tests; not for paper discovery or provenance decisions.
---

# Add a model

Use the flat `src/models/<module>/` layout. Models and methods are peers; do not create family directories.

1. Resolve the public model name, lowercase module slug, parameters, input needs,
   output type, and `upstream` or `rewrite` route.
2. Scaffold with `uv run tsf model add --name MyModel --params "enc_in:int,hidden:int=128"`. Add `--graph` only when dataset adjacency is consumed.
3. Run `uv run tsf component match <requirements> --json`, inspect promising candidates with `component show`, and list adapters before writing code. Retrieval is only a shortlist: reuse a component only after its mathematics, shapes, normalization, masking, residual order, initialization, and outputs match. Replace the placeholder in `model.py`; never import another named model package as an implementation dependency.
4. Complete `spec.py` with only the factory, parameter schema, config path, and
   runtime contract. Put descriptive and provenance facts in README front matter.
5. Complete the model card's method, structure, inputs/outputs, paper and codebase
   links, local implementation, differences, shared components, and constraints.
6. Run `uv run tsf model show MyModel`, `uv run tsf smoke --model MyModel`, and `uv run tsf repo doctor --forward`.

Success requires one indexed model card and runtime spec, a preset, passing
construction/forward/backward checks, and truthful provenance. Use
`port-upstream-model` for a licensed direct port or `rewrite-model-clean-room`
for an independent implementation before this integration step.

Use `curate-components` when the task is broader consolidation across existing
models; do not expand a single-model addition into an unsolicited refactor.
