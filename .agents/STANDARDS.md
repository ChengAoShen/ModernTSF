# Repository standards

Read only the section relevant to the current change. `AGENTS.md` contains the
always-on rules; this file keeps detailed contracts out of the default context.

## Models

Every model or method is a peer under `src/models/<lowercase_module_slug>/`;
there are no architecture categories or separate method hierarchy. Each entry
owns `model.py`, a `spec.py` containing its schema, factory, runtime contract,
provenance and verification record, and a checked `README.md` model card.

The catalog is the only discovery source. Configs are runnable presets, not
registrations. Runtime facts are capabilities, not categories. A releasable
entry must import, validate parameters, construct, pass its minimal forward and
backward contracts, return finite correctly shaped output, and support its
provenance claims with recorded evidence.

## Components

Reusable building blocks live in `src/components/`; they never classify
models. Extract code only when all consumers share mathematical behavior,
shapes, normalization, masking, residual order, initialization, and output
structure. Similar names are not evidence of equivalence.

Every extraction needs focused unit tests and affected-model contract tests.
Material variants remain local and explicitly named. Named model packages must
not import implementation code from peer models; proven shared code moves into
a cataloged component. Avoid catch-all utility modules and flag-driven base
classes that conceal paper-specific behavior.

## Adapters

Shared approximation backends live in `src/adapters/`, separate from named
models and paper-neutral components. Each has an `AdapterSpec` describing its
runtime contract and limitations. Consumers record the adapter key, use
`evidence="adaptation"`, and must not claim paper reproduction. Do not hide an
adapter under `src/models/` or present it as implementations of multiple named
algorithms.

## Paper and source evidence

Claims require evidence rather than inference from a model name. Record title,
authors, year, venue, persistent identifier, official source, pinned revision
or release, license, and inspected local files. Map defining paper operations
to code and record differences in preprocessing, architecture, loss, training,
output, and defaults.

Use one evidence label:

- `upstream-port`: traceable to a pinned upstream implementation.
- `reference-aligned`: independently implemented and numerically compared.
- `paper-reimplementation`: implemented from the paper without reference parity.
- `adaptation`: intentionally changes the source method.
- `unverified`: evidence is insufficient.

These labels are not model categories. Shape-only smoke tests cannot promote an
entry above `unverified`, and documentation must not overstate fidelity.

## Documentation ownership

Human documentation lives at the repository root, under `docs/`, and in model
cards. It explains public behavior and public CLI workflows without Agent paths,
prompt syntax, or internal command modules. Agent procedures live only under
`.agents/` and must not duplicate human tutorials. Schemas, specs, catalogs,
configs, and tests are executable truth; generated tables are projections.

Update code truth first, then regenerate or revise the human projection. Keep
English and Chinese page sets structurally mirrored. Do not hand-maintain facts
that can be rendered from a catalog.

## Skills

Skills live only at `.agents/skills/<skill-name>/SKILL.md`, with standard
kebab-case `name` and discriminating `description` frontmatter. Each skill owns
one recognizable outcome, expected inputs, preflight checks, execution path,
success criteria, artifacts, and stopping conditions. It uses public `tsf`
commands and contains no harness-specific paths, provider assumptions, retired
aliases, internal script entry points, or copies of human-facing tutorials.

Changed skills must pass `uv run python -m tsf_core.agent_assets` and the upstream
frontmatter validator. Test descriptions against positive, indirect,
incomplete, negative, and edge-case requests.
