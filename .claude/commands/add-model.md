Guide the user through adding a new model to the ModernTSF project.

Ask for the model name, then walk through these steps:

1. **Create** the model package at `src/models/<ModelName>/` with three files:
   - `model.py` — PyTorch `nn.Module` implementation
   - `schema.py` — Pydantic `ModelParameterConfig`
   - `registry.py` — `register()` function that calls `MODEL_REGISTRY.register(...)`

2. **Register** — add the model to `MODEL_NAME_MAP` in `src/benchmark/registry/models.py`:
   ```python
   MODEL_NAME_MAP["<ModelName>"] = "models.<model_name>.registry"
   ```

3. **Create** `configs/models/<ModelName>.toml`:
   ```toml
   [model]
   name = "<ModelName>"

   [model.params]
   enc_in = 7
   # ... other params
   ```

4. **Use** in a run config via `extends`.

Key rules:
- The model factory signature is `lambda cfg, params: model_instance` where `cfg` is the full `RootConfig`.
- `forward(self, x, x_mark, dec_inp, dec_mark)` — accept and ignore unused args with `*args` if the model doesn't use temporal marks.

Refer to `docs/en/add-model.md` for complete code examples.
