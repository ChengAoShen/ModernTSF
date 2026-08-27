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
| `MoFo` | https://github.com/PoorOtterBob/MoFo/tree/2d14b47ea839c3809952b412340d72393f2521dc | MIT |
| `PHAT` | https://github.com/PoorOtterBob/PHAT | to confirm |
| `BiST` | https://github.com/PoorOtterBob/BiST/tree/dd94adf7721fcbb9e3feb5d1b44040305199a4cc | no license declared; redistribution grant to confirm |
| `MAGE` | https://github.com/PoorOtterBob/MAGE/tree/f1fdd27da4e72a140c4f341f94d368fbcaec7507 | no license declared; redistribution grant to confirm |
| `STOP` | https://github.com/PoorOtterBob/STOP/tree/8babb610ece36a4215b2f66e1ef4a154f0c4f440 (under LargeST) | no license declared; redistribution grant to confirm |
| `CauAir` | https://github.com/PoorOtterBob/CauAir/tree/73dae00ca6ad14abb15174a0a0286d500e868b94 | no license declared; redistribution grant to confirm |
| `AirCade` | https://github.com/PoorOtterBob/AirCade/tree/179067f5b9fbc05f894022809e0b1c83e9f61fd8 | no license declared; redistribution grant to confirm |

Note: `src/models/phat/layers/PHAT_Attention.py` is **not** vendored — the
upstream never released it; it is an unverified reconstruction from the paper
(arXiv:2602.00654). See `docs/en/models.md`.

## CauAir air-quality models

Vendored from the [CauAir](https://github.com/PoorOtterBob/CauAir/tree/73dae00ca6ad14abb15174a0a0286d500e868b94) benchmark
(`src/models/<name>.py`), with `BaseModel` replaced by `nn.Module` and explicit
parameters. Several are CauAir's own re-implementations of published models; the
original references (where the upstream file declared one) are listed below.
License of the CauAir repository is **to confirm** (same status as the PR #2
PoorOtterBob set above).

| Model | Upstream | Original reference | License |
|---|---|---|---|
| `ASTGCN` | CauAir (`src/models/astgcn/_upstream.py`) | https://github.com/guoshnBJTU/ASTGCN-r-pytorch/tree/2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296 | no license declared |
| `GCLSTM` | CauAir (`src/models/gclstm/_upstream.py`) | no author source identified | no license declared |
| `DeepAir` | CauAir (`src/models/deepair/_upstream.py`) | no author source identified | no license declared |
| `STTN` | CauAir (`src/models/sttn/_upstream.py`) | https://github.com/xumingxingsjtu/STTN/tree/d24f8d331a6d81b819cfe0a9430793ae028d25ad (TensorFlow; local code differs materially) | no license declared |
| `GAGNN` | CauAir (`src/models/gagnn/_upstream.py`) | https://github.com/Friger/GAGNN/tree/509ac7d6eb55914979fc45f6d23e967021cfd270 (MIT; local code is an adaptation) | CauAir has no license declaration |
| `PM25_GNN` | CauAir (`src/models/pm25gnn/_upstream.py`) | https://github.com/shuowang-ai/PM2.5-GNN/tree/471fc60775f80492f4f224203d172868bc6eebac | MIT; local code is an adaptation |
| `AirFormer` | CauAir (`src/models/airformer/_upstream.py`) | https://github.com/yoshall/airformer/tree/ef7d3933768490e3a06921b8eb0f837c61741194 | no license declared |
| `DSTAGNN` | CauAir (`src/models/dstagnn/_upstream.py`) | https://github.com/SYLan2019/DSTAGNN/tree/10da1eb9e9d23412a83ea6ccc30b649da6402fba | no license declared |
| `PCDCNet` | CauAir (`src/models/pcdcnet/_upstream.py`) | no author source identified | no license declared |
| `AirPhyNet` | CauAir (`src/models/airphynet/_upstream.py`) | https://github.com/kethmih/AirPhyNet/tree/e77576cfea777e8cd07f2ae198c560a8790f4b91 | MIT |
| `AirDualODE` | CauAir (`src/models/airdualode/_upstream.py`) | https://github.com/decisionintelligence/Air-DualODE/tree/3accfef5d3ab40f685ea29f302f76287706ba821 | no license declared |
| `HL` | CauAir (`src/models/hl/_upstream.py`) | no associated paper or author source identified | no license declared |
| `LSTM` | CauAir (`src/models/lstm/_upstream.py`) | classic LSTM paper; local baseline comes from CauAir | no license declared |
| `RPMixer` | CauAir (`src/models/rpmixer/_upstream.py`) | no official paper code identified | no license declared |
| `MGSFformer` | CauAir (`src/models/mgsfformer/_upstream.py`) | https://github.com/GestaltCogTeam/MGSFformer/tree/ff665a422a0ae001cfdd1b60ec9b4338a5ab406e | no license declared |

A shared `src/components/graph_utils.py` (adjacency normalization helpers
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
| `Informer` | https://github.com/thuml/Time-Series-Library/tree/2fb5b84ecef67c45a759f7cf82023d27afe27882 | MIT |
| `Transformer` | https://github.com/thuml/Time-Series-Library/tree/2fb5b84ecef67c45a759f7cf82023d27afe27882 | MIT |
| `Reformer` | https://github.com/thuml/Time-Series-Library/tree/3a4819420d14095354aae96750ce8c499ef5f05e | MIT; local code is an adaptation |
| `Pyraformer` | https://github.com/thuml/Time-Series-Library/tree/3a4819420d14095354aae96750ce8c499ef5f05e | MIT |
| `ETSformer` | https://github.com/thuml/Time-Series-Library/tree/230805fe9f451b61e34b96116d995b417e343ac0 | MIT |
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
| `CATS` | https://github.com/dongbeank/CATS/tree/58854fc759d608ce400f378be83f4513960e505d | MIT |
| `Autoformer` | https://github.com/thuml/Autoformer/tree/51c7d416ae120b805fd5beef2f4ccf7de496a6ff | MIT; local code is an adaptation |
| `FEDformer` | https://github.com/MAZiqing/FEDformer/tree/c0f6b972def125691434d62be1ecadf710ae921a | MIT; local code is an adaptation |
| `PatchTST` | https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9 | Apache-2.0; local code is an adaptation |

## Recent 2025/2026 time-series model adapters

These entries register native ModernTSF implementations that follow the public
model names and high-level forecasting biases of verified open-source conference
work. The repository does not vendor those projects' training harnesses or
source files; the shared approximation backend lives in `src/adapters/recent_tsf.py`.
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
`src/adapters/ml_tsf.py`. They register familiar forecasting families under the
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
| `GWNet` | https://github.com/GestaltCogTeam/BasicTS/tree/c218c07b6ce5e4cf908b147fd180c486346fed9c/baselines/GWNet/arch | Apache-2.0 |
| `STGCN` | https://github.com/GestaltCogTeam/BasicTS/tree/79641b1c75246ab2d8c53bb52f2ac72588be0cdc/baselines/STGCN/arch | Apache-2.0 |
| `DCRNN` | Local port cites BasicTS (exact imported revision unresolved); official reference: https://github.com/liyaguang/DCRNN/tree/602afd30ddff5deed1e68f01828f3ff8f600131b | BasicTS Apache-2.0; official DCRNN MIT |
| `MTGNN` | Local port cites BasicTS (exact imported revision unresolved); official reference: https://github.com/nnzhan/MTGNN/tree/f811746fa7022ebf336f9ecd2434af5f365ecbf6 | BasicTS Apache-2.0; official MTGNN MIT |
| `AGCRN` | https://github.com/GestaltCogTeam/BasicTS (baselines/AGCRN/arch @79641b1) | Apache-2.0 |
| `STNorm` | https://github.com/GestaltCogTeam/BasicTS (baselines/STNorm/arch @79641b1) | Apache-2.0 |
| `StemGNN` | https://github.com/GestaltCogTeam/BasicTS (baselines/StemGNN/arch @79641b1) | Apache-2.0 |
| `STGODE` | https://github.com/GestaltCogTeam/BasicTS/tree/c218c07b6ce5e4cf908b147fd180c486346fed9c/baselines/STGODE/arch | Apache-2.0; local code is an adaptation |
| `STAEformer` | https://github.com/GestaltCogTeam/BasicTS (baselines/STAEformer/arch @79641b1) | Apache-2.0 |
| `GTS` | https://github.com/GestaltCogTeam/BasicTS/tree/79641b1/baselines/GTS/arch (gts_arch.py + gts_cell.py) | Apache-2.0 |
| `DGCRN` | Local port cites BasicTS (exact imported revision unresolved); official reference: https://github.com/FIBLAB/Traffic-Benchmark/tree/b9f8e8018480d36f58f790576f32e4157a76d3d4 | BasicTS Apache-2.0; official Traffic-Benchmark MIT |
| `STDN` | https://github.com/GestaltCogTeam/BasicTS/tree/c218c07b6ce5e4cf908b147fd180c486346fed9c/baselines/STDN/arch | Apache-2.0 |
| `DFDGCN` | https://github.com/GestaltCogTeam/DFDGCN/tree/3105058512a9279c000e98046a49d1baf3469884 | MIT |
| `STPGNN` | https://github.com/GestaltCogTeam/BasicTS/blob/da87bf443f285562341e7aaa3822825f399fe557/baselines/STPGNN/arch/stpgnn_arch.py | Apache-2.0 |
| `D2STGNN` | https://github.com/GestaltCogTeam/BasicTS/tree/79641b1c75246ab2d8c53bb52f2ac72588be0cdc/baselines/D2STGNN/arch | Apache-2.0 |
| `MegaCRN` | https://github.com/GestaltCogTeam/BasicTS/blob/79641b1/baselines/MegaCRN/arch/megacrn_arch.py | Apache-2.0 |
| `HimNet` | https://github.com/GestaltCogTeam/BasicTS/tree/c218c07b6ce5e4cf908b147fd180c486346fed9c/baselines/HimNet/arch | Apache-2.0 |
| `BigST` | https://github.com/GestaltCogTeam/BasicTS/tree/c218c07b6ce5e4cf908b147fd180c486346fed9c/baselines/BigST/arch | Apache-2.0 |
| `STWave` | https://github.com/GestaltCogTeam/BasicTS/blob/79641b1/baselines/STWave/arch/stwave_arch.py | Apache-2.0 |

## LatentTSF (added via PR #22)

| Model | Upstream | License |
|---|---|---|
| `LatentTSF` | https://github.com/Muyiiiii/LatentTSF | MIT |

The ModernTSF `latenttsf` implementation is a faithful **reimplementation** of the
two-stage algorithm (a frozen per-timestep MLP autoencoder + a DLinear backbone
forecasting in the latent space), not a verbatim copy of the upstream training
harness.
## TimeAlign (added via PR #23)

| Model | Upstream | License |
|---|---|---|
| `TimeAlign` | https://github.com/TROUBADOUR000/TimeAlign | to confirm |

The vendored `_TimeAlignCore` (plus the `Normalize` / `PositionalEmbedding` /
glocal-alignment layers) reproduces the upstream `Model` verbatim. The upstream
repository shipped no explicit license file at vendoring time (it builds on
Time-Series-Library, THUML); confirm redistribution terms with the authors.
## CRIB (added via PR #24)

| Model | Upstream | License |
|---|---|---|
| `CRIB` | https://github.com/Muyiiiii/CRIB | to confirm |

The vendored CRIB core (TCN + unified-variate Transformer + IB latent) reproduces
the upstream architecture (dead/unused submodules dropped). The upstream
repository shipped no explicit license file at vendoring time; confirm
redistribution terms with the authors. The missing-value data pipeline is not
vendored.
## GlocalIB (added via PR #25)

| Model | Upstream | License |
|---|---|---|
| `GlocalIB` | https://github.com/Muyiiiii/NeurIPS-25-Glocal-IB | MIT |

The vendored alignment wrapper plus the `cos_align` / `contrastive` losses are
pure PyTorch (no pypots/pygrinder). The ModernTSF model is a forecasting
adaptation of the upstream time-series-imputation regularizer.

## Shared utilities

| Utility | Upstream | License |
|---|---|---|
| `models/_external/adj_norm.py` (adjacency normalizations) | https://github.com/GestaltCogTeam/BasicTS (basicts/utils/adjacent_matrix_norm.py) — ported in spirit to dense numpy | Apache-2.0 |

<!-- Tier 1 / Tier 2 benchmark ports append their upstream + license here as they land. -->
