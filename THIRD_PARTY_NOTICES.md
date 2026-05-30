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

<!-- Tier 1 / Tier 2 benchmark ports append their upstream + license here as they land. -->
