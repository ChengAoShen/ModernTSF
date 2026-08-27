# Repository standards

Read only the section relevant to the current change. `AGENTS.md` contains the
always-on rules; this file keeps detailed contracts out of the default context.

## Models

Every model or method is a peer under `src/models/<lowercase_module_slug>/`;
there are no architecture categories or separate method hierarchy. Each entry
owns `model.py`, a checked `README.md` model card, and a `spec.py` limited to its
factory, parameter schema, config path, and runtime contract.

The catalog index is built from model-card front matter. Configs are runnable
presets, not registrations. Runtime facts are capabilities, not categories. A
releasable entry must import, validate parameters, construct, pass its declared
contracts, return finite correctly shaped output, and support its provenance.

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
models and paper-neutral components. Each has a runtime contract and disclosed
limitations. A consumer that materially changes a cited method is a `rewrite`
and must describe the adaptation in its model card. Do not hide an adapter under
`src/models/` or present it as several named algorithms.

## Model-card provenance

Each model `README.md` is the descriptive source of truth. Its front matter uses
only `implementation: upstream | rewrite` and records paper and codebase fields.
The body maps defining operations to code and states differences in preprocessing,
architecture, objective, training, output, and defaults.

Required front matter is `name`, `implementation`, `summary`, paper `title`,
`venue`, `year`, `url`, and codebase `url`, `revision`, `license`, `usage`. Empty
source facts must be explicit; do not omit keys or invent values.

Use `upstream` only for a direct port from an authoritative, licensed, pinned
revision after outputs, intermediate tensors, input and parameter gradients, and
train/eval behavior pass parity. Otherwise use `rewrite`, implemented from the
paper, textbook, or method description without copying unlicensed source. An
unlicensed repository may appear as `codebase.usage: reference-only`; state that
its code was not copied. A shape-only smoke test proves neither route. Do not keep
an unverified status or persist audit blockers as model metadata.

## Documentation ownership

Human documentation lives at the repository root, under `docs/`, and in model
cards. It explains public behavior and public CLI workflows without Agent paths,
prompt syntax, or internal command modules. Agent procedures live only under
`.agents/` and must not duplicate human tutorials. Schemas, model-card front
matter, runtime specs, configs, and tests are executable truth; generated indexes
and tables are projections.

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
