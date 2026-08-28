---
name: port-upstream-model
description: Port a forecasting model directly from an authoritative upstream repository with a compatible verified license and pinned revision. Use when source tracing and numerical fidelity are required; not for unlicensed, ambiguous, or paper-only implementations.
---

# Port a licensed upstream model

Before copying code, verify the source is authoritative, the exact commit or release
is pinned, and its license permits the intended redistribution and modification.
Record relevant paths and preserve required copyright, license, and attribution.
Stop if any of these facts is unknown; route paper-based work to
`rewrite-model-clean-room` without using the source implementation.

Map upstream files, symbols, parameters, buffers, tensor axes, preprocessing,
defaults, initialization, output semantics, and train/eval branches to the local
package. Keep the smallest faithful change needed for ModernTSF's runtime contract.
Do not hide material alterations behind compatibility wrappers.

Set README front matter to `implementation: upstream` only after recording:

- codebase URL, full revision, SPDX-compatible license, and `usage: ported`;
- local-to-upstream file and parameter mapping;
- required notices and all behavioral differences;
- shared components whose equivalence has been demonstrated.

Use `verify-upstream-parity` before accepting the route. Then run the model audit,
forward/backward contracts, state-dict checks, and repository audit. If parity
cannot be reached, repair the port or replace it with an actual clean-room rewrite;
changing the label alone is invalid.
