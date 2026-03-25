Run a benchmark experiment for this project.

Ask the user which config file they want to run (e.g. `configs/runs/run_single_data.toml`, `configs/runs/sweep_model.toml`, `configs/runs/sweep_data.toml`, `configs/runs/multi_sweep.toml`), then execute:

```bash
uv run modern-tsf --config <config_path>
```

If they haven't specified a config, help them find the right one:
- Single dataset + model: `configs/runs/run_single_data.toml`
- Sweep over models: `configs/runs/sweep_model.toml`
- Sweep over datasets: `configs/runs/sweep_data.toml`
- Multi-axis sweep: `configs/runs/multi_sweep.toml`

After the run, offer to aggregate results:

```bash
uv run python tool/aggregate_results.py --dataset <dataset_name>
```
