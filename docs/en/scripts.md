# Tooling (`tsf`) & scripts

ModernTSF exposes a single unified entry point — `tool/tsf.py` — that an agent (or
you) can drive for every common operation. It is pure standard library
(`argparse` + `concurrent.futures` + `subprocess`), runs through `uv`, and is
concurrent where it helps. The old `run_multi_configs.sh` and
`aggregate_and_plot.sh` glue scripts have been retired in favour of `tsf run` and
`tsf aggregate-plot`.

```bash
uv run python tool/tsf.py <command> [args...]
uv run python tool/tsf.py --help                 # list all commands
uv run python tool/tsf.py <command> --help       # a command's own flags
```

---

## Scaffold

| Command | Purpose |
|---|---|
| `new-model` | Scaffold a model package + `schema.py` + `registry.py` + config + smoke run config, and insert the `MODEL_NAME_MAP` entry. |
| `new-dataset` | Scaffold a dataset (`--pattern custom` / `presplit` = config only; `single` = full code + wiring). |

```bash
# A plain (B, T, C) forecaster with two hyper-parameters
uv run python tool/tsf.py new-model --name MyModel --params "enc_in:int,hidden:int=128"

# A node-structured graph / spatiotemporal model (reads params["adj_mx"])
uv run python tool/tsf.py new-model --name MyGraphNet --graph --params "enc_in:int,hidden:int=64"

# A config-only custom CSV dataset
uv run python tool/tsf.py new-dataset --name my_csv --pattern custom \
    --root-path ./dataset/my_csv --data-path my_csv.csv --target OT
```

After `new-model`, fill the architecture into the generated `model.py` `forward`,
then verify (below).

---

## Verify & run (concurrent)

| Command | Purpose |
|---|---|
| `smoke` | Run smoke config(s) end-to-end and report PASS/FAIL. `--all`, `--model <Name>`, or `--config <paths>`; `--jobs N` (default `min(8, cpu)`). |
| `run` | Run experiment config(s). `--jobs N` (default 1) for parallel runs; `--gpus 0,1` round-robins `CUDA_VISIBLE_DEVICES` across jobs. |

```bash
# Verify a single new model
uv run python tool/tsf.py smoke --model MyModel

# Verify every model in the repo, 8 at a time
uv run python tool/tsf.py smoke --all --jobs 8

# Run two sweep configs in parallel across two GPUs
uv run python tool/tsf.py run configs/runs/sweep_data.toml configs/runs/sweep_model.toml --jobs 2 --gpus 0,1
```

`smoke` exits non-zero if any config fails, so it doubles as a CI gate.

---

## Results & plots

| Command | Purpose |
|---|---|
| `report` | Generate a shareable Markdown report (leaderboard + bubble chart + results table). `--dataset`, `--pred-len`, `--top`, `--out`, `--no-plot`. |
| `aggregate-plot` | Aggregate a dataset's results + render a bubble chart in one shot (replaces `aggregate_and_plot.sh`). `--dataset`, `--pred-len`, `--x/--y/--size`, `--out-csv/--out-svg`. |
| `aggregate` | → `tool/aggregate_results.py` |
| `rank` | → `tool/rank_models.py` |
| `plot` | → `tool/plot_bubble.py` |
| `characteristics` | → `tool/dataset_characteristics.py` |
| `visualize` | → `tool/visual_data.py` |
| `predictions` | → `tool/visualize_predictions.py` |
| `inspect` | → `tool/inspect_config.py` |

```bash
uv run python tool/tsf.py aggregate-plot --dataset ETTh1 --pred-len 96
uv run python tool/tsf.py rank --dataset ETTh1
uv run python tool/tsf.py inspect --config configs/runs/multi_sweep.toml
```

---

## Data prep

| Command | Purpose |
|---|---|
| `pre-process` | → `tool/pre_process.py` (CSV → pre-windowed `.npz`) |
| `convert-traffic` | → `tool/convert_traffic.py` (value array + adjacency → node bundle) |
| `gift-download` | → `tool/gift_eval_download.py` (download GIFT-EVAL datasets) |

The forwarding commands accept the underlying tool's flags verbatim, so
`tsf aggregate --dataset ETTh1` is exactly `python tool/aggregate_results.py --dataset ETTh1`.

---

## `scripts/detect_hardware.sh`

The one remaining shell script: detect the GPU / driver / CUDA version and
recommend a uv PyTorch backend tag for `UV_TORCH_BACKEND`
(`cpu | cu118 | cu121 | cu124 | cu126 | cu128`). Used by the `setup-env` skill;
see [setup-env.md](setup-env.md).

```bash
bash scripts/detect_hardware.sh             # human-readable report
bash scripts/detect_hardware.sh --backend   # print only the backend tag
UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync --python 3.12
```

Output:

```
gpu=NVIDIA GeForce RTX 4090
driver=550.54.15
cuda=12.4
backend=cu124
```

- No GPU / no `nvidia-smi` on `PATH` → reports `backend=cpu`.
- Maps the driver's max CUDA version to the highest available wheel backend ≤ that version.
- Read-only: it never installs anything — it only reports and recommends.
