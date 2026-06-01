# Third-Party Notices

ModernTSF vendors model implementations ported from external research
repositories. Each lives under `src/models/<name>/` with a `_upstream.py`
(verbatim upstream code, import paths adjusted) or equivalent, plus a thin
ModernTSF adapter. The vendored code remains subject to its **original upstream
license**; the module docstrings cite the upstream source URL. Where a project
ships no explicit license, redistribution status should be confirmed with the
upstream authors before relying on it.

> ⚠️ The licenses below are recorded per upstream repository. Entries marked
> *"to confirm"* had no clearly declared license at vendoring time and should be
> verified with the authors.

## PoorOtterBob models (added via PR #2)

| Model | Upstream | License |
|---|---|---|
| `MoFo` | https://github.com/PoorOtterBob/MoFo | to confirm |
| `PHAT` | https://github.com/PoorOtterBob/PHAT | to confirm |
| `BiST` | https://github.com/PoorOtterBob/BiST | to confirm |
| `MAGE` | https://github.com/PoorOtterBob/MAGE | to confirm |
| `STOP` | https://github.com/PoorOtterBob/STOP (under LargeST) | to confirm |
| `CauAir` | https://github.com/PoorOtterBob/CauAir | to confirm |
| `AirCade` | https://github.com/PoorOtterBob/AirCade | to confirm |

Note: `src/models/phat/layers/PHAT_Attention.py` is **not** vendored — the
upstream never released it; it is an unverified reconstruction from the paper
(arXiv:2602.00654). See `docs/en/models.md`.

## CauAir air-quality models

Vendored from the [CauAir](https://github.com/PoorOtterBob/CauAir) benchmark
(`src/models/<name>.py`), with `BaseModel` replaced by `nn.Module` and explicit
parameters. Several are CauAir's own re-implementations of published models; the
original references (where the upstream file declared one) are listed below.
License of the CauAir repository is **to confirm** (same status as the PR #2
PoorOtterBob set above).

| Model | Upstream | Original reference | License |
|---|---|---|---|
| `ASTGCN` | CauAir (src/models/astgcn.py) | https://github.com/guoshnBJTU/ASTGCN-r-pytorch | to confirm |
| `GCLSTM` | CauAir (src/models/gclstm.py) | — | to confirm |
| `DeepAir` | CauAir (src/models/deepair.py) | — | to confirm |
| `STTN` | CauAir (src/models/sttn.py) | — | to confirm |
| `GAGNN` | CauAir (src/models/gagnn.py) | GAGNN (torch_scatter/geometric deps replaced with pure PyTorch) | to confirm |
| `PM25_GNN` | CauAir (src/models/pm25_gnn.py) | — | to confirm |
| `AirFormer` | CauAir (src/models/airformer.py) | — | to confirm |
| `DSTAGNN` | CauAir (src/models/dstagnn.py) | — | to confirm |
| `PCDCNet` | CauAir (src/models/pcdcnet.py) | — | to confirm |
| `AirPhyNet` | CauAir (src/models/airphynet.py) | — | to confirm |
| `AirDualODE` | CauAir (src/models/airdualode.py) | — | to confirm |
| `HL` | CauAir (src/models/hl.py) | — | to confirm |
| `LSTM` | CauAir (src/models/lstm.py) | — | to confirm |
| `RPMixer` | CauAir (src/models/rpmixer.py) | — | to confirm |
| `MGSFformer` | CauAir (src/models/mgsfformer.py) | — | to confirm |
| `CATS` | CauAir (src/models/cats.py) | — | to confirm |

A shared `src/models/_external/graph_utils.py` (adjacency normalization helpers
used by the graph adapters) accompanies these models.

## Tier 1 / benchmark ports

| Model | Upstream | License |
|---|---|---|
| `TimeXer` | https://github.com/thuml/Time-Series-Library | MIT |
| `Crossformer` | https://github.com/thuml/Time-Series-Library | MIT |
| `MICN` | https://github.com/thuml/Time-Series-Library | MIT |
| `FiLM` | https://github.com/thuml/Time-Series-Library | MIT |
| `Koopa` | https://github.com/thuml/Time-Series-Library | MIT |
| `FreTS` | https://github.com/thuml/Time-Series-Library | MIT |
| `ModernTCN` | https://github.com/thuml/Time-Series-Library | MIT |
| `Informer` | https://github.com/thuml/Time-Series-Library | MIT |
| `Transformer` | https://github.com/thuml/Time-Series-Library | MIT |
| `Reformer` | https://github.com/thuml/Time-Series-Library | MIT |
| `Pyraformer` | https://github.com/thuml/Time-Series-Library | MIT |
| `ETSformer` | https://github.com/thuml/Time-Series-Library | MIT |
| `NSTransformer` | https://github.com/thuml/Time-Series-Library | MIT |
| `SOFTS` | https://github.com/thuml/Time-Series-Library | MIT |
| `WPMixer` | https://github.com/thuml/Time-Series-Library | MIT |
| `MultiPatchFormer` | https://github.com/thuml/Time-Series-Library | MIT |
| `PAttn` | https://github.com/thuml/Time-Series-Library/blob/main/models/PAttn.py | MIT |
| `CARD` | https://github.com/wxie9/CARD/blob/main/long_term_forecast_l96/models/CARD.py | No explicit LICENSE in upstream wxie9/CARD; built on Time-Series-Library (TSLib), MIT |
| `Fredformer` | https://github.com/chenzRG/Fredformer | No explicit LICENSE in upstream (KDD 2024 research code, "Fredformer") — to confirm |
| `DUET` | https://github.com/decisionintelligence/DUET | MIT (Copyright (c) 2024 Huawei Technologies Co., Ltd) |
| `TimeKAN` | https://github.com/huangst21/TimeKAN | Apache-2.0 |
| `MTSMixer` | https://github.com/plumprc/MTS-Mixers/blob/main/models/MTSMixer.py | No license declared in upstream plumprc/MTS-Mixers (no LICENSE file; GitHub license API returns 404; README has no license notice) — to confirm |
| `UMixer` | https://github.com/XiangMa-Shaun/U-Mixer/blob/main/models/UMixer.py | No LICENSE file in upstream XiangMa-Shaun/U-Mixer (AAAI 2024); built on Time-Series-Library (TSLib, MIT) but upstream provides no explicit license — to confirm |
| `Pathformer` | https://github.com/decisionintelligence/pathformer | NOASSERTION (no LICENSE file declared in upstream; official ICLR 2024 code release) — to confirm |
| `NHiTS` | https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nhits.py | Apache-2.0 |
| `NBeats` | https://github.com/philipperemy/n-beats | MIT |
| `WaveNet` | https://github.com/GestaltCogTeam/BasicTS/blob/v0.5.8/baselines/WaveNet/arch.py | Apache-2.0 |
| `DeepAR` | https://github.com/GestaltCogTeam/BasicTS/blob/79641b1c75246ab2d8c53bb52f2ac72588be0cdc/baselines/DeepAR/arch/deepar_arch.py | Apache-2.0 |
| `DSFormer` | https://github.com/GestaltCogTeam/DSformer | No license declared (GitHub license API returns null; no LICENSE file; README has no license notice) — all rights reserved by authors. Not GPL/AGPL. Original ChengqingYu/DSformer redirects here. |
| `Sumba` | https://github.com/chenxiaodanhit/Sumba | No license file (GitHub license API returns null) — all rights reserved by authors. Not GPL/AGPL. |
| CrossGNN | https://github.com/hqh0728/CrossGNN | No explicit upstream license (all rights reserved) — to confirm |
| `HDMixer` | https://github.com/hqh0728/HDMixer | No LICENSE file in upstream (GitHub license API returns 404; all rights reserved). `layers/box_coder1D.py` carries a permissive Facebook/Meta copyright header. Not GPL/AGPL. |
| `SRSNet` | https://github.com/decisionintelligence/SRSNet | MIT (Copyright (c) 2024 Huawei Technologies Co., Ltd) |
| `DTAF` | https://github.com/decisionintelligence/DTAF | No explicit LICENSE file in upstream; published by decisionintelligence as an AAAI'26 baseline inside the MIT-licensed TFB benchmark (https://github.com/decisionintelligence/TFB). Treated as MIT-compatible via parent TFB; not GPL/AGPL. |
| `TimePerceiver` | https://github.com/efficient-learning-lab/TimePerceiver | MIT |
| `MambaSimple` | https://github.com/thuml/Time-Series-Library/blob/main/models/MambaSimple.py | MIT |
| `MSGNet` | https://github.com/thuml/Time-Series-Library/blob/main/models/MSGNet.py | MIT |
| `TimeFilter` | https://github.com/TROUBADOUR000/TimeFilter | No explicit LICENSE file (GitHub API reports license: null); README acknowledges Time-Series-Library (MIT) and iTransformer (MIT) as the codebases it derives from. Not GPL/AGPL/copyleft. |
| `S_Mamba` | https://github.com/wzhwzhwzh0921/S-D-Mamba (model/S_Mamba.py, layers/Mamba_EncDec.py) | No explicit LICENSE file in upstream S-D-Mamba repo; core is iTransformer inverted embedding + Mamba, treated as MIT per author/TSLib provenance. Kernel-free Mamba reused from MIT thuml/Time-Series-Library MambaSimple via src/models/mambasimple. |
| `BiMamba` | https://github.com/Huangmr0719/BiMamba (BiMamba.py) | No license declared (unlicensed; license: None per GitHub API). Not GPL/AGPL, so not skipped. The kernel-free selective scan it uses is borrowed from src/models/mambasimple (MIT, thuml/Time-Series-Library + mamba-minimal). |
| `S4` | https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py | Apache-2.0 |

## Recent 2025/2026 time-series model adapters

These entries register native ModernTSF implementations that follow the public
model names and high-level forecasting biases of verified open-source conference
work. The repository does not vendor those projects' training harnesses or
source files; the shared implementation lives in `src/models/_recent_tsf.py`.
Use the upstream repositories below for paper-specific reproduction claims.

| Model | Venue/source tag | Upstream reference | License |
|---|---|---|---|
| `Aurora` | ICLR 2026 | https://github.com/decisionintelligence/Aurora | to confirm |
| `TimeAlign` | ICLR 2026 | https://github.com/TROUBADOUR000/TimeAlign | to confirm |
| `GTR` | ICLR 2026 | https://github.com/macovaseas/GTR | to confirm |
| `PhaseFormer` | ICLR 2026 | https://github.com/neumyor/PhaseFormer_TSL | to confirm |
| `PMDformer` | ICLR 2026 | https://github.com/aohu1105/PMDformer | to confirm |
| `MMPD` | ICLR 2026 | https://github.com/Thinklab-SJTU/MMPD | to confirm |
| `COSA` | ICLR 2026 | https://github.com/bigbases/COSA_ICLR2026 | to confirm |
| `DistDF` | ICLR 2026 | https://github.com/Master-PLC/DistDF | to confirm |
| `Sonnet` | AAAI 2026 | https://github.com/ClaudiaShu/Sonnet | to confirm |
| `APN` | AAAI 2026 | https://github.com/decisionintelligence/APN | to confirm |
| `TimeCAP` | AAAI 2026 | https://github.com/RCR-LYY/TimeCAP | to confirm |
| `GOTSF` | AAAI 2026 | https://github.com/netop-team/gotsf | to confirm |
| `FTP` | AAAI 2026 | https://github.com/Zhveh7/FTP | to confirm |
| `OccamVTS` | AAAI 2026 | https://github.com/sisuolv/OccamVTS | to confirm |
| `HN_MVTS` | AAAI 2026 | https://github.com/av-savchenko/HN-MVTS | to confirm |
| `SEMPO` | NeurIPS 2025 | https://github.com/mala-lab/SEMPO | to confirm |
| `InterPDN` | AAAI 2026 | https://github.com/leonardokong486/interPDN | to confirm |
| `TimeO1` | NeurIPS 2025 | https://github.com/Master-PLC/Time-o1 | to confirm |
| `FeTS` | AAAI 2026 | https://github.com/lllucky111/FeTS | to confirm |
| `SymTime` | NeurIPS 2025 | https://github.com/wwhenxuan/SymTime | to confirm |
| `ImplicitForecaster` | NeurIPS 2025 | https://github.com/rakuyorain/Implicit-Forecaster | to confirm |
| `AMRC` | NeurIPS 2025 | https://github.com/MazelTovy/AMRC | to confirm |
| `HMformer` | AAAI 2026 | https://github.com/dantian123121/HMformer | to confirm |
| `TiRex` | NeurIPS 2025 | https://github.com/NX-AI/tirex | to confirm |
| `LatentTSF` | ICML 2026 | https://github.com/Muyiiiii/LatentTSF | to confirm |
| `CoRA` | ICLR 2026 | https://github.com/decisionintelligence/CoRA | to confirm |
| `DynamicTMoE` | ICML 2026 | https://github.com/andone-07/Dynamic-TMoE | to confirm |
| `PULSE` | ICML 2026 | https://github.com/Gemost/PULSE | to confirm |
| `OLinear` | NeurIPS 2025 | https://github.com/jackyue1994/OLinear | to confirm |
| `MAFS` | NeurIPS 2025 | https://github.com/h505023992/MAFS | to confirm |
| `TSRAG` | NeurIPS 2025 | https://github.com/UConn-DSIS/TS-RAG | to confirm |
| `TimeMosaic` | AAAI 2026 | https://github.com/BenchCouncil/TimeMosaic | to confirm |
| `Kronos` | AAAI 2026 | https://github.com/shiyu-coder/Kronos | to confirm |

## Classical ML / statistical time-series adapters

These entries are native ModernTSF PyTorch implementations in
`src/models/_ml_tsf.py`. They register familiar forecasting families under the
standard time-series interface so the normal trainer can move them to CPU,
CUDA, or MPS. ModernTSF does **not** vendor source code from XGBoost, LightGBM,
CatBoost, statsmodels, scikit-learn, or other upstream classical ML packages
for these adapters.

| Model family | Registered models | Implementation |
|---|---|---|
| Linear regularized regression | `RidgeRegressionTS`, `LassoRegressionTS`, `ElasticNetTS`, `BayesianRidgeTS`, `PolynomialRegressionTS` | Torch lag-window heads plus differentiable regularization |
| Kernel / prototype regression | `KNNForecasterTS`, `SVRForecasterTS`, `GaussianProcessTS` | Trainable prototypes with RBF weighting |
| Tree and boosting style ensembles | `DecisionTreeTS`, `RandomForestTS`, `ExtraTreesTS`, `GradientBoostingTS`, `XGBoostTS`, `LightGBMTS`, `CatBoostTS` | Differentiable soft-tree ensembles over lag features |
| Statistical forecasters | `ARIMATS`, `AutoRegressiveTS`, `ExpSmoothingTS`, `KalmanFilterTS` | Differentiable ARIMA-like, smoothing, and alpha-beta update modules |
| Basic neural baselines | `MLPForecasterTS`, `RNNForecasterTS`, `GRUForecasterTS`, `LSTMForecasterTS`, `TCNForecasterTS` | Small Torch neural forecasters |

## Tier 2 / graph models

| Model | Upstream | License |
|---|---|---|
| `STID` | https://github.com/GestaltCogTeam/BasicTS (src/basicts/models/STID/arch/stid_arch.py) | Apache-2.0 |
| `GWNet` | https://github.com/GestaltCogTeam/BasicTS (baselines/GWNet/arch/gwnet_arch.py) | Apache-2.0 |
| `STGCN` | https://github.com/GestaltCogTeam/BasicTS/tree/79641b1c75246ab2d8c53bb52f2ac72588be0cdc/baselines/STGCN/arch | Apache-2.0 |
| `DCRNN` | https://github.com/GestaltCogTeam/BasicTS (baselines/DCRNN/arch @79641b1) | Apache-2.0 |
| `MTGNN` | https://github.com/GestaltCogTeam/BasicTS (baselines/MTGNN/arch @79641b1) | Apache-2.0 |
| `AGCRN` | https://github.com/GestaltCogTeam/BasicTS (baselines/AGCRN/arch @79641b1) | Apache-2.0 |
| `STNorm` | https://github.com/GestaltCogTeam/BasicTS (baselines/STNorm/arch @79641b1) | Apache-2.0 |
| `StemGNN` | https://github.com/GestaltCogTeam/BasicTS (baselines/StemGNN/arch @79641b1) | Apache-2.0 |
| `STGODE` | https://github.com/GestaltCogTeam/BasicTS (baselines/STGODE/arch @79641b1) | Apache-2.0 |
| `STAEformer` | https://github.com/GestaltCogTeam/BasicTS (baselines/STAEformer/arch @79641b1) | Apache-2.0 |
| `GTS` | https://github.com/GestaltCogTeam/BasicTS/tree/79641b1/baselines/GTS/arch (gts_arch.py + gts_cell.py) | Apache-2.0 |
| `DGCRN` | https://github.com/GestaltCogTeam/BasicTS (baselines/DGCRN/arch @79641b1) | Apache-2.0 |
| `STDN` | https://github.com/GestaltCogTeam/BasicTS/tree/v0.5.8/baselines/STDN/arch | Apache-2.0 |
| `DFDGCN` | https://github.com/GestaltCogTeam/DFDGCN/blob/main/DFDGCN/basicts/archs/arch_zoo/dfdgcn_arch/dfdgcn_arch.py (official reference impl by the BasicTS authors; not present in BasicTS baselines/ at 79641b1) | MIT |
| `STPGNN` | https://github.com/GestaltCogTeam/BasicTS/blob/da87bf443f285562341e7aaa3822825f399fe557/baselines/STPGNN/arch/stpgnn_arch.py | Apache-2.0 |
| `D2STGNN` | https://github.com/GestaltCogTeam/BasicTS/tree/79641b1/baselines/D2STGNN/arch | Apache-2.0 |
| `MegaCRN` | https://github.com/GestaltCogTeam/BasicTS/blob/79641b1/baselines/MegaCRN/arch/megacrn_arch.py | Apache-2.0 |
| `HimNet` | https://github.com/GestaltCogTeam/BasicTS/tree/dev/next_generation/baselines/HimNet/arch | Apache-2.0 |
| `BigST` | https://github.com/GestaltCogTeam/BasicTS/tree/c218c07b6ce5e4cf908b147fd180c486346fed9c/baselines/BigST/arch | Apache-2.0 |
| `STWave` | https://github.com/GestaltCogTeam/BasicTS/blob/79641b1/baselines/STWave/arch/stwave_arch.py | Apache-2.0 |

## Shared utilities

| Utility | Upstream | License |
|---|---|---|
| `models/_external/adj_norm.py` (adjacency normalizations) | https://github.com/GestaltCogTeam/BasicTS (basicts/utils/adjacent_matrix_norm.py) — ported in spirit to dense numpy | Apache-2.0 |

<!-- Tier 1 / Tier 2 benchmark ports append their upstream + license here as they land. -->
