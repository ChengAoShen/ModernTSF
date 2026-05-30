# GIFT-EVAL benchmark

ModernTSF natively supports the [GIFT-EVAL](https://huggingface.co/datasets/Salesforce/GiftEval) benchmark — 53 dataset configurations spanning 23 base datasets, 10 frequencies (from secondly to monthly), and multiple domains (energy, traffic, weather, finance, and more).

---

## Step 1 — Download or link data

### Download all 53 datasets

```bash
uv run python tool/gift_eval_download.py --output-dir /your/path
```

Downloads from `Salesforce/GiftEval` on HuggingFace Hub into `--output-dir`, then creates a symlink at `./dataset/gift_eval` pointing to that directory. Default location is `~/.cache/gift_eval`.

### Download specific datasets only

```bash
uv run python tool/gift_eval_download.py --output-dir /your/path --datasets electricity/15T ett1/H m4_monthly
```

Pass any subset of the 53 dataset names. Use `--list` to print all valid names:

```bash
uv run python tool/gift_eval_download.py --list
```

### Link an already-downloaded directory

If the data is already on disk, skip downloading and only create the symlink:

```bash
uv run python tool/gift_eval_download.py --link-only --output-dir /path/to/existing/gift_eval
```

---

## Step 2 — Run the sweep

```bash
uv run modern-tsf --config configs/runs/gift_eval_sweep.toml
```

`gift_eval_sweep.toml` extends the base config and `DLinear.toml`, then uses `[sweep.extend]` to iterate all 53 dataset TOMLs in `configs/datasets/gift_eval/`. Replace `DLinear.toml` with any model config or add a `[sweep]` key to sweep multiple models.

Preview the expansion before running:

```bash
uv run python tool/inspect_config.py --config configs/runs/gift_eval_sweep.toml
```

---

## Prediction length tiers

GIFT-EVAL defines three prediction length tiers for each dataset. Each config file in `configs/datasets/gift_eval/` sets `pred_len` to the **short** term by default and documents the other two in a comment:

```toml
# pred_len follows GIFT-EVAL "short" term (1x).
# Other terms:  medium = 480 (10x),  long = 720 (15x)

[task]
pred_len = 48
```

To run medium or long term, update `pred_len` in the individual dataset TOML (or override it in your run config via `[sweep]`).

---

## Dataset list

| Dataset | Frequencies |
|---|---|
| `electricity` | 15T, D, H, W |
| `ett1`, `ett2` | 15T, D, H, W |
| `solar` | 10T, D, H, W |
| `LOOP_SEATTLE` | 5T, D, H |
| `M_DENSE` | D, H |
| `SZ_TAXI` | 15T, H |
| `bitbrains_fast_storage` | 5T, H |
| `bitbrains_rnd` | 5T, H |
| `bizitobs_application` | — |
| `bizitobs_l2c` | 5T, H |
| `bizitobs_service` | — |
| `hierarchical_sales` | D, W |
| `kdd_cup_2018_with_missing` | D, H |
| `saugeenday` | D, M, W |
| `us_births` | D, M, W |
| `m4_daily`, `m4_hourly`, `m4_monthly` | — |
| `m4_quarterly`, `m4_weekly`, `m4_yearly` | — |
| `car_parts_with_missing` | — |
| `covid_deaths` | — |
| `hospital` | — |
| `jena_weather` | — |
| `restaurant` | — |
| `temperature_rain_with_missing` | — |

---

## Script arguments

| Argument | Default | Description |
|---|---|---|
| `--output-dir DIR` | `~/.cache/gift_eval` | Download destination / symlink target |
| `--datasets NAME...` | all 53 | Specific datasets to download |
| `--link-only` | off | Skip download; only create symlink |
| `--list` | off | Print all 53 dataset names and exit |

---

## Notes

- The symlink is created at `./dataset/gift_eval`. All dataset TOMLs use `root_path = "./dataset/gift_eval"` so they work without further configuration.
- If `./dataset/gift_eval` already exists as a regular directory (not a symlink), the script warns and skips symlink creation.
- Already-downloaded datasets are detected by the presence of `dataset_info.json` and skipped automatically.
- `huggingface_hub` must be installed (`uv sync` includes it).
