# CLAUDE.md

This file provides guidance to coding agents (Claude Code, Codex, etc.) when working with code in this repository. `AGENTS.md` is a symlink to this file, and `.agents/skills/` links to `.claude/skills/`, so all agents share the same instructions and skills.

## Environment & Commands

`tool/tsf.py` is the unified Agent entry point for every tool (scaffold / smoke /
run / aggregate-plot + forwards to all `tool/*.py`). See the "Unified tooling
(`tsf`)" section below and `docs/en/scripts.md`. Quickest paths:
`uv run python tool/tsf.py new-model --name X --params "enc_in:int"` then
`uv run python tool/tsf.py smoke --model X`.

```bash
# Install / sync dependencies (Python 3.12). The PyTorch build (CPU vs CUDA) is
# chosen at install time via UV_TORCH_BACKEND — let uv auto-detect the GPU.
# See the `setup-env` skill / docs/en/setup-env.md. `bash scripts/detect_hardware.sh`
# reports the recommended backend.
UV_TORCH_BACKEND=auto uv sync --python 3.12   # or cu124 / cu121 / cpu …

# Run an experiment
uv run modern-tsf --config configs/runs/run_single_data.toml

# Preview config expansion (sweep counts, covered datasets/models)
uv run python tool/inspect_config.py --config configs/runs/multi_sweep.toml

# Aggregate performance + profile CSVs for a dataset
uv run python tool/aggregate_results.py --dataset ETTh1

# Plot a bubble chart from aggregated results
uv run python tool/plot_bubble.py --csv work_dirs/ETTh1/results_all.csv --x mse --y mae --size total_params

# Rank models per pred_len/seed
uv run python tool/rank_models.py --dataset ETTh1

# Visualise dataset samples
uv run python tool/visual_data.py --config configs/datasets/etth1.toml --split train --num-samples 3

# Extract TFB-style dataset characteristics (trend/seasonality/stationarity/...)
uv run python tool/dataset_characteristics.py --config configs/datasets/etth1.toml --split train --per-channel

# Plot forecast vs ground-truth case studies for a trained run
uv run python tool/visualize_predictions.py --config <cfg> --num-samples 4   # forecast vs ground-truth case plots

# Build a traffic/spatiotemporal node bundle (value array + adjacency) for cauair_st
uv run python tool/convert_traffic.py \
    --values dataset/metr_la/metr_la.npz --values-key data \
    --adj dataset/metr_la/adj_mx.pkl --output-dir dataset/metr_la \
    --seq-len 12 --pred-len 12 --add-time --freq-min 5 --splits 0.7,0.1,0.2

# Download GIFT-EVAL benchmark datasets from HuggingFace (for the gift_eval dataset)
uv run python tool/gift_eval_download.py

# Pre-process a CSV into pre-windowed .npz files for the pre_processed dataset
uv run python tool/pre_process.py \
    --input-csv dataset/ETT-small/ETTh1.csv \
    --output-dir dataset/ETTh1_npy \
    --seq-len 512 --label-len 0 --pred-len 96 --features M
```

There are no automated tests and no linting config. All source packages live under `src/`, which is the package root (`package-dir = {"" = "src"}`), so `import benchmark`, `import data`, and `import models` all resolve from there.

## Available Datasets

| Config | Name key | Description |
|---|---|---|
| `configs/datasets/etth1.toml` | `ETTh1` | ETT hourly dataset 1 |
| `configs/datasets/etth2.toml` | `ETTh2` | ETT hourly dataset 2 |
| `configs/datasets/ettm1.toml` | `ETTm1` | ETT minute dataset 1 |
| `configs/datasets/ettm2.toml` | `ETTm2` | ETT minute dataset 2 |
| `configs/datasets/electricity.toml` | `electricity` | Electricity consumption |
| `configs/datasets/weather.toml` | `weather` | Weather multivariate |
| `configs/datasets/traffic.toml` | `traffic` | Road traffic |
| `configs/datasets/solar.toml` | `solar` | Solar power (text file) |
| `configs/datasets/pre_processed.toml` | `pre_processed` | Pre-windowed .npz files |
| `configs/datasets/synthetic_st.toml` | `synthetic_st` | Synthetic node-structured (spatiotemporal mode) |
| `configs/datasets/cauair_ccaq_st.toml` | `cauair_st` | CauAir/CCAQ air quality, node layout (spatiotemporal / covariate) |
| `configs/datasets/cauair_ccaq_ts.toml` | `cauair_ts` | CauAir/CCAQ data flattened to channels (forecasting) |
| `configs/datasets/exchange.toml` | `custom` | Exchange-rate (plain CSV) |
| `configs/datasets/ili.toml` | `custom` | Influenza-like illness (plain CSV) |
| `configs/datasets/beijing_air.toml` | `custom` | Beijing air quality (plain CSV) |
| `configs/datasets/aqshunyi.toml` | `custom` | AQShunyi air quality (plain CSV) |
| `configs/datasets/aqwan.toml` | `custom` | AQWan air quality (plain CSV) |
| `configs/datasets/nn5.toml` | `custom` | NN5 cash-withdrawals (plain CSV) |
| `configs/datasets/fred_md.toml` | `custom` | FRED-MD macroeconomic (plain CSV) |
| `configs/datasets/metr_la.toml` | `cauair_st` | METR-LA traffic graph (node + adjacency) |
| `configs/datasets/pems_bay.toml` | `cauair_st` | PEMS-BAY traffic graph (node + adjacency) |
| `configs/datasets/pems03.toml` | `cauair_st` | PEMS03 traffic graph (node + adjacency) |
| `configs/datasets/pems04.toml` | `cauair_st` | PEMS04 traffic graph (node + adjacency) |
| `configs/datasets/pems07.toml` | `cauair_st` | PEMS07 traffic graph (node + adjacency) |
| `configs/datasets/pems08.toml` | `cauair_st` | PEMS08 traffic graph (node + adjacency) |
| `configs/datasets/gift_eval/*.toml` | `gift_eval` | GIFT-EVAL benchmark datasets (53 configs; see `gift-eval` skill / `docs/en/gift-eval.md`) |

The generic `name = "custom"` key wires any plain flat-multivariate CSV through `Dataset_Custom` with no new code — only a config. Traffic graph bundles (`metr_la`/`pems_bay`/`pems0x`) reuse the `cauair_st` node loader; see `docs/en/datasets-traffic.md`.

Synthetic datasets (`periodic`, `trend`) have source code under `src/data/datasets/` but no config file by default — create `configs/datasets/<name>.toml` as needed.

## Task modes

All tasks are forecasting; `task.mode` selects the data setting (default `time_series`):

- `time_series` — `(B, T, C)` value batches; every channel is a target. Unchanged historical behaviour.
- `spatiotemporal` — node-structured datasets return `(value (T,N), value (T,N), cov (T,N,F), cov (T,N,F))`; models rebuild `(B, T, N, 1+F)`. Target is the value of all `N` nodes.
- `covariate` — like spatiotemporal but the model also receives the future covariate block `(B, pred_len, N, F)`.

Model adapters are polymorphic on the mark rank (`src/models/_external/marks.py`): 3-D `(B,T,6)` = raw calendar stamps; 4-D `(B,T,N,F)` = node covariates. See `docs/en/task-modes.md`.

## Available Models (172)

Models are grouped into categories; counts below. Each model has a config at
`configs/models/<Name>.toml` and source under `src/models/<name>/`.

| Category | Count |
|---|---|
| Linear-based | 6 |
| Transformer-based | 21 |
| MLP / Patch-based | 11 |
| CNN-based | 5 |
| RNN-based | 6 |
| Modern forecasters | 10 |
| Architecture variants | 4 |
| Filter-based | 2 |
| Other | 7 |
| Ported PoorOtterBob | 7 |
| Graph / Spatiotemporal (Tier 2) | 20 |
| CauAir air-quality (graph + baselines) | 16 |
| Classic ML / statistical baselines | 24 |
| Recent TSF + foundation models | 33 |
| **Total** | **172** |

See `docs/en/models.md` for the full per-model table.

## Architecture

### Config → Registry → Runner pipeline

The CLI entry point (`src/benchmark/cli.py`) drives everything in three steps:

1. **Load configs** — `benchmark.config.load_config(path)` reads a TOML file, resolves `extends` chains via deep-merge, expands `[sweep]` and `[sweep.extend]` into a cartesian product, and validates each expanded dict against `RootConfig` (Pydantic). Returns a list of `LoadedConfig` objects.

2. **Register components** — `register_from_config(config)` lazily imports and calls `register()` on the dataset, model, and metric modules referenced by name. Registration is idempotent (tracked in module-level sets). The maps that drive this are `DATASET_NAME_MAP`, `MODEL_NAME_MAP` in their respective registry modules.

3. **Run sweep** — `run_sweep(configs)` iterates the list and calls `run_one(config, raw, sweep_keys)` for each. `run_one` builds three DataLoaders (train/val/test), constructs the model, trains with early stopping, evaluates, and writes a CSV summary row to `work_dirs/<dataset>/<model>/performance.csv`.

### Registry pattern

Every extensible component (datasets, models, metrics, losses) uses the same pattern:
- A `*_REGISTRY` singleton in `src/benchmark/registry/`
- A `*_NAME_MAP` dict mapping string name → dotted module path
- Each module exposes a `register()` function that calls `REGISTRY.register(name, cls_or_factory, schema)`
- Schemas are Pydantic models that validate `dataset.params` / `model.params` from the TOML

### Data loading

`src/data/provider.py::build_data_loader` is the single factory used by `run_one`. It looks up the dataset class by name, passes `root_path`, `data_path`, `size=(seq_len, label_len, pred_len)`, `flag`, `features`, and unpacked `dataset_params` to the constructor.

Three dataset patterns exist:
- **Single-file** (`ForecastingDataset` subclass): inherits `_get_borders` for ratio-based splitting, implements `_read_data`. Examples: `Dataset_Custom`, `Dataset_ETT_hour`, `Dataset_Solar`.
- **Pre-split** (`Dataset_PreSplit`, direct `Dataset` subclass): reads `train.csv`/`val.csv`/`test.csv` from `root_path`. Use `name = "presplit"` and `data_path = ""` in config. Scaler is always fitted on `train.csv`.
- **Pre-processed** (`Dataset_PreProcessed`): reads pre-windowed `.npz` files produced by `tool/pre_process.py`. Use `name = "pre_processed"` and `data_path = ""`.

`__getitem__` always returns `(input_series, output_series, input_stamp, output_stamp)` as float32 numpy arrays. Timestamps are always real arrays (zero-filled when no `date` column), never `None`, to keep PyTorch's collate safe.

### Model interface

All models receive `(x, x_mark, dec_inp, dec_mark)` from `_call_model` in `trainer.py`. Models that don't use temporal marks should accept `*args` in `forward`. The factory lambda in each `registry.py` receives `(cfg: RootConfig, params: dict)`.

### Config inheritance

TOML files compose via `extends = [list of paths]` resolved relative to the file. Merge order: earlier files in the list are base, later files override. The current file overrides everything. `[sweep.extend]` named axes load entire TOML files as extra override layers, expanding via cartesian product before the regular `[sweep]` keys expand.

## Key files for extending the project

| Task | File(s) |
|---|---|
| Add a dataset (single-file) | `src/data/datasets/<name>.py`, `src/data/schemas/datasets/<name>.py`, `src/benchmark/registry/datasets.py` (`DATASET_NAME_MAP`), `configs/datasets/<name>.toml` |
| Add a dataset (pre-split) | `configs/datasets/<name>.toml` only — use `name = "presplit"` |
| Add a dataset (pre-processed .npz) | Run `tool/pre_process.py`, then `configs/datasets/<name>.toml` with `name = "pre_processed"` |
| Add a model | `src/models/<name>/model.py`, `schema.py`, `registry.py`, `src/benchmark/registry/models.py` (`MODEL_NAME_MAP`), `configs/models/<name>.toml` |
| Add a metric or loss | `src/benchmark/registry/metrics.py` or `losses.py` |
| Change training loop | `src/benchmark/runner/trainer.py` |
| Change evaluation | `src/benchmark/runner/evaluator.py` |
| Change config schema | `src/benchmark/config/schema/` |

## Agent Skills

`.claude/skills/` is the single agent-facing entry layer (Codex and other agents discover the same skills via the `.agents/skills` symlink). Each skill wraps the underlying `tool/` scripts and `scripts/` shell helpers so they stay the single source of truth — invoke skills instead of re-deriving CLI calls. The old `.claude/commands/` has been replaced by these skills.

| Skill | Wraps |
|---|---|
| `setup-env` | hardware detection + `UV_TORCH_BACKEND` uv install (`scripts/detect_hardware.sh`) |
| `run` | `modern-tsf --config …` (single/sweep runs; mentions `[training.tricks]` + `[evaluation] strategy="rolling"`) |
| `experiments` | one-click ablation / hyperparameter sweeps (`[sweep]`) + forecast case plots (`tool/visualize_predictions.py`); wraps `docs/en/experiments.md` |
| `characteristics` | `tool/dataset_characteristics.py` (trend/seasonality/stationarity stats) |
| `aggregate` | `tool/aggregate_results.py` (+ TFB fairness `--collapse`/`--null-threshold`; optional bubble plot) |
| `visualize` | `tool/visual_data.py` (dataset samples; points to `experiments` for prediction plots) |
| `pre-process` | `tool/pre_process.py` |
| `add-dataset` | `tool/tsf.py new-dataset` (custom/presplit/single scaffold) |
| `add-model` | `tool/tsf.py new-model` scaffold + `tsf smoke` verify |
| `understand-model` | model `README.md` cards as progressive disclosure (paper venue/date/arXiv/abstract → source on demand) |
| `smoke` | `tool/tsf.py smoke` (concurrent end-to-end PASS/FAIL verification) |
| `inspect` | `tool/inspect_config.py` |
| `rank` | `tool/rank_models.py` (+ TFB fairness `--null-threshold`/`--aggregate`/`--fill-nan-with-mean`) |
| `plot` | `tool/plot_bubble.py` |
| `report` | `tool/tsf.py report` (Markdown report: leaderboard + bubble chart + table) |
| `gift-eval` | GIFT-EVAL download + 53-dataset sweep |
| `sweep` | `tool/tsf.py run` (concurrent multi-config runs) |
| `submit` | `tool/tsf.py trace` / `submit` / `leaderboard-build` (TSEval submission flow) |
| `report-issue` | `gh issue create` / `gh pr create` against `Diaugeia/ModernTSF` (upstream defect reporting) |

If you discover a defect in ModernTSF itself while working (crash in `src/` or
`tool/`, wrong shapes, config/doc mismatch), use the `report-issue` skill: ask
the user whether to file a GitHub issue or open a PR upstream — never file
without asking.

## Unified tooling (`tsf`)

`tool/tsf.py` is the single entry point for every tool — pure standard library
(`argparse` + `concurrent.futures` + `subprocess`), run via `uv`, concurrent
where it helps. It replaces the retired `run_multi_configs.sh` /
`aggregate_and_plot.sh` glue. Full reference: `docs/en/scripts.md`.

```bash
uv run python tool/tsf.py --help                 # list all commands

# Scaffold (one command -> package + config + smoke config + registry entry)
uv run python tool/tsf.py new-model --name MyModel --params "enc_in:int,hidden:int=128"
uv run python tool/tsf.py new-model --name MyGraphNet --graph     # graph/spatiotemporal
uv run python tool/tsf.py new-dataset --name my_csv --pattern custom --root-path ./dataset/my_csv

# Verify end-to-end, concurrently (doubles as a CI gate; non-zero on any failure)
uv run python tool/tsf.py smoke --model MyModel
uv run python tool/tsf.py smoke --all --jobs 8

# Run experiment config(s) concurrently (replaces run_multi_configs.sh)
uv run python tool/tsf.py run configs/runs/sweep_model.toml --jobs 2 --gpus 0,1

# Aggregate a dataset's results + bubble chart (replaces aggregate_and_plot.sh)
uv run python tool/tsf.py aggregate-plot --dataset ETTh1 --pred-len 96

# Generate a shareable Markdown report (leaderboard + bubble chart + table)
uv run python tool/tsf.py report --dataset ETTh1
```

`tsf` also forwards verbatim to every `tool/*.py`: `report`, `aggregate`, `rank`,
`plot`, `characteristics`, `visualize`, `predictions`, `inspect`, `pre-process`,
`convert-traffic`, `gift-download`.

## TSEval contract & submission

ModernTSF is the *producer* side of the TSEval leaderboard. The contract layer
`src/tsf_core/` (pydantic only, no torch) defines `DatasetSpec` / `RunRecord` /
`SubmissionReport` and exports them to `src/tsf_core/schema/*.json` — the only
artifact the TSEval consumer reads. The `schema-check` GitHub Action runs
`schema-export --check` on every PR, so changing a model without re-exporting
fails CI.

```bash
# Export / verify the JSON Schema contract
uv run python tool/tsf.py schema-export            # regenerate schema/*.json
uv run python tool/tsf.py schema-export --check    # fail if committed schema is stale

# Capture an agent's experiment process (CLI-boundary, agent-agnostic)
uv run python tool/tsf.py trace start [--label L]  # begin a trajectory session
uv run python tool/tsf.py trace status             # inspect the active session
uv run python tool/tsf.py trace end                # close it

# Package a finished run into a Submission Report (then PR the bundle to Diaugeia/TSEval)
uv run python tool/tsf.py submit --dataset ETTh1 --model DLinear --latest

# Aggregate submissions into a ranked leaderboard.json (consumer side; no torch)
uv run python tool/tsf.py leaderboard-build --source work_dirs/_submissions --out leaderboard.json
```

The `leaderboard-build` step reads every `<id>/submission.json` under a
submissions root, checks each (result + trajectory present, schema-valid), and
collates them into one `leaderboard.json` ranked per (track, dataset, horizon).
That JSON is what the TSEval Hugging Face Space (`Diaugeia/TSEval`, a self-
contained static frontend) renders.

Every run also writes a self-describing
`work_dirs/<dataset>/<model>/records/<run_id>.json` (a validated `RunRecord`). A
submission bundles that result + a `trajectory.jsonl` + a small report; the full
flow is in `docs/en/tseval-submit.md`.

## Scripts

The only remaining shell script is `scripts/detect_hardware.sh` — report GPU /
driver / CUDA and recommend a `UV_TORCH_BACKEND` tag (`--backend` prints only the
tag). Used by the `setup-env` skill.

```bash
bash scripts/detect_hardware.sh           # human-readable report
UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync --python 3.12
```

## Detailed docs

- `docs/en/` — English reference (params, configs, add-dataset, add-model, tools)
- `docs/zh-CN/` — Chinese mirror (same content, kept in sync)

Key doc files (full index: `docs/en/README.md` / `docs/zh-CN/README.md`):
- Environment setup (GPU/CUDA): `docs/en/setup-env.md`
- Parameters reference: `docs/en/params.md`
- Config loading and usage: `docs/en/configs.md`
- Models reference: `docs/en/models.md`
- Task modes (data settings, forecasting-only scope): `docs/en/task-modes.md`
- Add a new model: `docs/en/add-model.md`
- Add a new dataset: `docs/en/add-dataset.md`
- Traffic / spatiotemporal graph bundles: `docs/en/datasets-traffic.md`
- Pre-process datasets: `docs/en/pre-process.md`
- Experiments (sweeps + forecast case plots): `docs/en/experiments.md`
- Inspect config expansion: `docs/en/inspect-config.md`
- Visualize datasets: `docs/en/visualize-data.md`
- Dataset characteristics (TFB-style): `docs/en/dataset-characteristics.md`
- Aggregate results: `docs/en/aggregate-results.md`
- Model rankings: `docs/en/rank-models.md`
- Bubble chart: `docs/en/plot-bubble.md`
- GIFT-EVAL benchmark: `docs/en/gift-eval.md`
- Workflow shell scripts: `docs/en/scripts.md`
