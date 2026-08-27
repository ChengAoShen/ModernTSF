# Public command line

ModernTSF has one public entry point:

```bash
uv run tsf --help
```

The commands below are the supported user interface.

## Models and repository

```bash
uv run tsf model add --name MyModel --params "enc_in:int,hidden:int=128"
uv run tsf model list
uv run tsf model show DLinear
uv run tsf model audit DLinear
uv run tsf component match "patch transformer normalization"
uv run tsf repo audit
uv run tsf repo doctor --forward
uv run tsf repo doctor --backward
```

`component match` returns candidates that still require semantic review. `repo audit` checks repository consistency; `repo doctor` additionally constructs every model. `--forward` executes registered tensor contracts and regression seeds, while `--backward` also checks finite training gradients and implies `--forward`.

## Data

```bash
uv run tsf dataset add --help
uv run tsf dataset prepare --help
uv run tsf dataset inspect --help
uv run tsf dataset plot --help
uv run tsf dataset convert-traffic --help
uv run tsf dataset gift-download --help
```

## Experiments

```bash
uv run tsf inspect --config configs/runs/run_single_data.toml
uv run tsf run configs/runs/run_single_data.toml
uv run tsf run configs/runs/a.toml configs/runs/b.toml --jobs 2 --gpus 0,1
uv run tsf smoke --model DLinear
uv run tsf smoke --all --jobs 8
```

## Results and submissions

```bash
uv run tsf result aggregate --help
uv run tsf result rank --help
uv run tsf result plot --help
uv run tsf result predictions --help
uv run tsf result report --help
uv run tsf trace start --label experiment
uv run tsf submit --dataset ETTh1 --model DLinear --latest
uv run tsf schema-export --check
uv run tsf leaderboard-build --source work_dirs/_submissions
```

Use `uv run tsf <group> <operation> --help` for operation-specific flags.

## Hardware detection

`scripts/detect_hardware.sh` is a read-only helper for choosing `UV_TORCH_BACKEND`:

```bash
bash scripts/detect_hardware.sh
UV_TORCH_BACKEND=auto uv sync --python 3.12
```
