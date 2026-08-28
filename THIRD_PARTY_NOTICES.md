# Third-Party Notices

ModernTSF distinguishes two implementation origins in each model card's YAML
front matter:

- `implementation: upstream` identifies code ported from a named upstream
  repository at a pinned revision. The upstream license continues to govern
  the ported material.
- `implementation: rewrite` identifies an independent implementation written
  for this repository. A paper or codebase URL in a rewrite model card is a
  research reference, not a statement that the referenced source code is
  vendored or copied.

The model card at `src/models/<model>/README.md` is the canonical record for
each model's provenance, paper, codebase URL, pinned revision, license label,
and usage. This notice is a compact index of the 29 models currently declared
as upstream ports. License labels below reproduce the corresponding model-card
metadata; consult the linked upstream repository and revision for the complete
license text and notices.

## Upstream ports

| Model | Upstream repository | Pinned revision | License | Usage |
|---|---|---|---|---|
| `AGCRN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `CATS` | [dongbeank/CATS](https://github.com/dongbeank/CATS) | `58854fc759d608ce400f378be83f4513960e505d` | `MIT` | `ported` |
| `CycleNet` | [ACAT-SCUT/CycleNet](https://github.com/ACAT-SCUT/CycleNet) | `d807e51fc2dcd143885ee639d97965a7ab0926f4` | `Apache-2.0` | `ported` |
| `D2STGNN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `79641b1c75246ab2d8c53bb52f2ac72588be0cdc` | `Apache-2.0` | `ported` |
| `DFDGCN` | [GestaltCogTeam/DFDGCN](https://github.com/GestaltCogTeam/DFDGCN) | `3105058512a9279c000e98046a49d1baf3469884` | `MIT` | `ported` |
| `DLinear` | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `Apache-2.0` | `ported` |
| `ETSformer` | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | `230805fe9f451b61e34b96116d995b417e343ac0` | `MIT` | `ported` |
| `FITS` | [VEWOXIC/FITS](https://github.com/VEWOXIC/FITS) | `d040bb015b6299da26d879b90dd19c80fb72c160` | `Apache-2.0` | `ported` |
| `GWNet` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `HimNet` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `Informer` | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | `2fb5b84ecef67c45a759f7cf82023d27afe27882` | `MIT` | `ported` |
| `Linear` | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `Apache-2.0` | `ported` |
| `MoFo` | [PoorOtterBob/MoFo](https://github.com/PoorOtterBob/MoFo) | `2d14b47ea839c3809952b412340d72393f2521dc` | `MIT` | `ported` |
| `NBeats` | [philipperemy/n-beats](https://github.com/philipperemy/n-beats) | `06a4e209ada80bf1f403ced5228261784dfb26ed` | `MIT` | `ported` |
| `NHiTS` | [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) | `6c4f3e557d0ed672314323edba972eb550cb3550` | `Apache-2.0` | `ported` |
| `NLinear` | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `Apache-2.0` | `ported` |
| `PaiFilter` | [aikunyi/FilterNet](https://github.com/aikunyi/FilterNet) | `cdb321c4e338e0c07b45cee92f54b3c5bd5a809e` | `Apache-2.0` | `ported` |
| `SegRNN` | [lss-1138/SegRNN](https://github.com/lss-1138/SegRNN) | `8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f` | `Apache-2.0` | `ported` |
| `SparseTSF` | [lss-1138/SparseTSF](https://github.com/lss-1138/SparseTSF) | `b8c2740eecc84d8095ffce49ba5acafe68e53bb8` | `Apache-2.0` | `ported` |
| `STAEformer` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `STDN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `StemGNN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `STGCN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `STID` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `STNorm` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `TexFilter` | [aikunyi/FilterNet](https://github.com/aikunyi/FilterNet) | `cdb321c4e338e0c07b45cee92f54b3c5bd5a809e` | `Apache-2.0` | `ported` |
| `TimeBridge` | [Hank0626/TimeBridge](https://github.com/Hank0626/TimeBridge) | `0f9a83fbc3e1260c9ddd527c522dff0ce4b9554b` | `MIT` | `ported` |
| `TimeKAN` | [huangst21/TimeKAN](https://github.com/huangst21/TimeKAN) | `3a7c366a9e8547fd8840c5d27f25ee3e30615e33` | `Apache-2.0` | `ported` |
| `Transformer` | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | `2fb5b84ecef67c45a759f7cf82023d27afe27882` | `MIT` | `ported` |

## Independent rewrites and references

The other 149 registered models are declared as `implementation: rewrite`.
Their model cards record the papers and, where useful, external repositories
consulted as references. Those links do not add the referenced repositories to
ModernTSF and do not change the provenance of the local independent
implementations. Consult each model card for its implementation notes and any
documented differences from the cited paper or reference codebase.
