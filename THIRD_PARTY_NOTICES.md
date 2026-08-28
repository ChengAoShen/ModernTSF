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
and usage. This notice is a compact index of the 13 models currently declared
as upstream ports. License labels below reproduce the corresponding model-card
metadata; consult the linked upstream repository and revision for the complete
license text and notices.

## Upstream ports

| Model | Upstream repository | Pinned revision | License | Usage |
|---|---|---|---|---|
| `AGCRN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `D2STGNN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `79641b1c75246ab2d8c53bb52f2ac72588be0cdc` | `Apache-2.0` | `ported` |
| `DFDGCN` | [GestaltCogTeam/DFDGCN](https://github.com/GestaltCogTeam/DFDGCN) | `3105058512a9279c000e98046a49d1baf3469884` | `MIT` | `ported` |
| `GWNet` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `HimNet` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `NBeats` | [philipperemy/n-beats](https://github.com/philipperemy/n-beats) | `06a4e209ada80bf1f403ced5228261784dfb26ed` | `MIT` | `ported` |
| `NHiTS` | [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) | `6c4f3e557d0ed672314323edba972eb550cb3550` | `Apache-2.0` | `ported` |
| `STAEformer` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `STDN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `StemGNN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `STGCN` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `STID` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |
| `STNorm` | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) | `c218c07b6ce5e4cf908b147fd180c486346fed9c` | `Apache-2.0` | `ported` |

## Independent rewrites and references

The other 165 registered models are declared as `implementation: rewrite`.
Their model cards record the papers and, where useful, external repositories
consulted as references. Those links do not add the referenced repositories to
ModernTSF and do not change the provenance of the local independent
implementations. Consult each model card for its implementation notes and any
documented differences from the cited paper or reference codebase.
