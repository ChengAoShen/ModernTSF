# Add a model or method

Models and methods are peers under `src/models/<module>/`. Do not introduce architecture-family directories. The catalog points to one `spec.py` per public name; presets configure runs but do not register models.

## Scaffold

```bash
uv run tsf model add --name MyModel \
  --params "enc_in:int,hidden:int=128,dropout:float=0.1"
```

Use `--graph` only when the model consumes dataset adjacency. The command creates:

```text
src/models/my_model/
  model.py
  spec.py
  README.md
configs/models/MyModel.toml
configs/runs/smoke_my_model.toml
```

It also adds the lazy `MODEL_CATALOG` reference.

## ModelSpec

`spec.py` is the single source of truth for:

- the Pydantic `ModelParameterConfig` and factory;
- public identity, config, model card, and smoke case;
- capabilities and output type;
- shared component dependencies;
- minimal task dimensions and regression seeds used by the executable contract.

The model README front matter is the descriptive and provenance source of truth
for paper, codebase, license, implementation route, and summary.

The model factory receives the resolved root config and validated `model.params`. Public forward input is `(x_enc, x_mark_enc, x_dec, x_mark_dec)`; point models return `(B, pred_len, C or N)`, while quantile and distribution models declare their extra output axis through capabilities.

Graph and calendar input conversion should reuse `components.marks`; reusable paper-neutral blocks belong in `src/components/`. Paper-specific operations remain in the model package.

## Evidence

A new entry is incomplete until it has executable upstream parity or clean-room rewrite evidence. Record every material difference in the model card; a correct output shape is necessary but is not reproduction evidence.

## Verify

```bash
uv run tsf model show MyModel
uv run tsf smoke --model MyModel
uv run tsf repo doctor --forward
```

All three must agree with the preset and model card before release.
