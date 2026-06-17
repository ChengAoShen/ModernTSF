# Changelog

All notable changes to ModernTSF are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semantic versioning.

## [0.3.3] — 2026-06-17

A bug-fix release.

### Fixed

- **`inverse=true` no longer crashes in `MS` mode.**
  `ForecastingDataset.inverse_transform` broadcast-failed when the model emits a
  single target column (`MS` = multivariate-input, single-target) while the
  scaler was fit on all `C` channels (`non-broadcastable operand (N,1) vs
  (N,C)`). A reduced-channel output is now anchored on the last `k` channels'
  statistics (the target channel(s), which time-series datasets place last);
  full-width `M`-mode inversion is byte-identical. Added
  `configs/runs/smoke_dlinear_msinv.toml` as a regression smoke.

## [0.3.0] — 2026-05-31

An agent-automation release: **99 → 115 models**, a single unified `tsf` tooling
entry point with concurrent scaffold / verify / run, one-shot Markdown reports,
and end-to-end smoke coverage for every model.

### Added

- **16 new CauAir air-quality models (99 → 115)** ported from
  [PoorOtterBob/CauAir](https://github.com/PoorOtterBob). These are the
  non-duplicate subset of the original 31-model port — the ~15 models that main
  had already ported independently from BasicTS (GWNet, STGCN, STID, DCRNN,
  DGCRN, STGODE, CrossGNN, D2STGNN, SOFTS, STNorm, STAEformer, TimeXer, UMixer,
  DSFormer, Pathformer) were skipped to avoid duplicate implementations.
  - Graph spatiotemporal: ASTGCN, GCLSTM, DeepAir, STTN, GAGNN, PM25_GNN,
    AirFormer, DSTAGNN, PCDCNet, AirPhyNet, AirDualODE
  - Non-graph baselines / forecasters: HL, LSTM, RPMixer, MGSFformer, CATS
  - Adapters wire each model to the `(cfg, params)` factory + `params["adj_mx"]`
    injection convention; a shared `src/models/_external/graph_utils.py` provides
    adjacency helpers.
- **Full smoke coverage (115/115).** Every model now has a
  `configs/runs/smoke_*.toml`; the 31 classic base models that previously lacked
  one were added, so `tsf smoke --all` is a concurrent end-to-end CI gate over
  the whole model zoo.

### Tooling — unified `tsf` Agent entry point

- **`tool/tsf.py`** — one standard-library entry point (no extra deps, run via
  `uv`, concurrent where it helps) for every tool:
  - `new-model` / `new-dataset` — one-command scaffold (package + schema +
    registry + config + smoke config + `MODEL_NAME_MAP` entry). `--graph` emits a
    spatiotemporal variant.
  - `smoke` — concurrent end-to-end PASS/FAIL verification (`--all` / `--model` /
    `--config`, `--jobs N`); non-zero exit on any failure.
  - `run` — concurrent multi-config runner (`--jobs`, `--gpus`).
  - `aggregate-plot` — aggregate + bubble chart in one shot.
  - `report` — generate a shareable Markdown report (leaderboard + bubble chart +
    results table) for a dataset.
  - forwards verbatim to every other `tool/*.py`.
- New `smoke` and `report` Agent Skills; `add-model` / `add-dataset` / `sweep`
  skills now lead with the `tsf` scaffold.
- Retired the `run_multi_configs.sh` and `aggregate_and_plot.sh` shell scripts
  (replaced by `tsf run` / `tsf aggregate-plot`); only `detect_hardware.sh` remains.

### Dependencies

- Added `torchdiffeq` (ODE solvers for AirPhyNet / AirDualODE).
- Declared `reformer-pytorch` (Reformer) and `pywavelets` (STWave) in
  `pyproject.toml` — these were previously imported but never declared, so a
  clean `uv sync` would prune them and break those models.

### Docs

- Repo-wide consistency/accuracy pass across README (en+zh), `docs/` (en+zh),
  CLAUDE.md, and the skills; model count updated to 115.
- Slimmed the README badge row to `Python 3.12+ · uv · PyTorch 2.6 · Time Series
  Forecasting · Models 100+ · License MIT`.

**Full diff:** https://github.com/Diaugeia/ModernTSF/compare/v0.2.0...v0.3.0

## [0.2.0] — 2026-05-30

A large expansion release: **31 → 99 models**, three forecasting data settings,
a graph-adjacency pipeline, new metrics/losses/training tricks, and a full
docs + Agent-Skills refresh. The project is now **forecasting-only** at every
reachable code path.

### Added

- **68 new models (31 → 99)** ported from BasicTS (Apache-2.0), Time-Series-Library
  (MIT), and TFB, plus the PoorOtterBob set. Highlights:
  - Transformers: TimeXer, Crossformer, Informer, Transformer, Reformer, Pyraformer,
    ETSformer, NSTransformer, MultiPatchFormer, PAttn, CARD, Fredformer, DUET,
    Pathformer, DSFormer, DTAF, TimePerceiver
  - MLP/Patch: WPMixer, MTSMixer, UMixer, NHiTS, NBeats, HDMixer, SRSNet
  - CNN: MICN, ModernTCN, WaveNet
  - RNN/SSM: DeepAR, MambaSimple, S_Mamba, BiMamba, S4 (kernel-free, CPU-runnable)
  - Modern: FiLM, FreTS, Koopa, SOFTS, TimeKAN
  - 20 graph/spatiotemporal (Tier 2): STID, GWNet, STGCN, DCRNN, MTGNN, AGCRN,
    STNorm, StemGNN, STGODE, STAEformer, GTS, DGCRN, STDN, DFDGCN, STPGNN, D2STGNN,
    MegaCRN, HimNet, BigST, STWave
  - PoorOtterBob: MoFo, PHAT, BiST, MAGE, STOP, CauAir, AirCade
- **Three forecasting data settings** via `task.mode`: `time_series`,
  `spatiotemporal`, `covariate` (polymorphic mark handling in
  `src/models/_external/marks.py`).
- **Graph adjacency pipeline** — datasets expose `adj_mx`/`num_nodes`; the runner
  injects them into the model factory. Optional `[dataset.params] adj_norm`
  (symmetric/scaled Laplacian, GCN, transition matrices).
- **Datasets** — node-structured CauAir/synthetic_st, traffic graph bundles
  (METR-LA, PEMS-BAY, PEMS03/04/07/08 via `tool/convert_traffic.py`), and plain-CSV
  configs (Exchange, ILI, Beijing-air, AQShunyi, AQWan, NN5, FRED-MD).
- **Metrics** — `corr`, `rse`, `wape`, `smape` (+ opt-in `mase`); **masked losses**
  (`masked_mae`/`mse`/`rmse`) for missing-value forecasting.
- **Pluggable training callbacks** — `[training.tricks]` curriculum / grad-clip /
  grad-accum, plus model aux-loss support.
- **`[evaluation] strategy="rolling"`** RollingForecast evaluation.
- **Profiling** — fit-time / inference-time columns in `performance.csv`.
- **Tools** — `tool/convert_traffic.py`, `tool/visualize_predictions.py`,
  `tool/dataset_characteristics.py`.
- **Agent Skills** — `experiments`, `characteristics` skills added (14 total),
  all with progressive disclosure (L1 frontmatter → L2 body → L3 docs/tools).
- **Governance** — MIT `LICENSE` (Copyright © 2026 Diaugeia.AI),
  `THIRD_PARTY_NOTICES.md` (per-model upstream + license for all 99), GitHub issue
  forms (new model / bug / feature), PR template, `CONTRIBUTING.md`.

### Changed

- README/README_zh add a **Principles** section (Modern / Agentic / Reproducible /
  Open by default) and a License footer; full accuracy + en↔zh parity pass across
  all docs.
- Copyright re-declared to **Diaugeia.AI**.

### Removed

- **Stripped all non-forecasting branches** (imputation / anomaly detection /
  classification) and the `task_name` parameter from 7 TSLib-style models
  (Autoformer, FEDformer, TimesNet, TiDE, SegRNN, CrossLinear, MoFo). The project
  is now forecasting-only.
- Dropped M4 and imputation-as-a-task from the roadmap (out of scope).

### Notes

- `S_Mamba` / `BiMamba` / `S4` use a kernel-free selective scan (no `mamba_ssm`),
  so they run on CPU; GPU runs are recommended for speed.
- `PHAT` ships a paper reconstruction of its attention block (not an
  author-verified reproduction) — see `THIRD_PARTY_NOTICES.md`.

**Full diff:** https://github.com/Diaugeia/ModernTSF/compare/v0.1.0...v0.2.0

## [0.1.0]

Initial release — 31 models, TOML-driven config/registry/runner pipeline, classic
benchmark datasets, aggregation/ranking/plotting tools, and Agent Skills.
