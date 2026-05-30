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

<!-- Tier 1 / Tier 2 benchmark ports append their upstream + license here as they land. -->
