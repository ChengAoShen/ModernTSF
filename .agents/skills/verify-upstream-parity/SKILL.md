---
name: verify-upstream-parity
description: Build and run numerical parity checks between a ModernTSF model and its licensed pinned upstream implementation. Use to qualify or revalidate the upstream route; not for paper-only structure validation or ordinary smoke tests.
---

# Verify upstream parity

Use an isolated checkout of the exact recorded revision and capture Python,
framework, dependency, device, dtype, and deterministic settings. Build an explicit
parameter and buffer mapping; never compare separately initialized models.

With identical loaded weights and inputs, compare:

- final outputs and selected defining intermediate tensors;
- input gradients and every active parameter gradient;
- train and eval behavior, including stochastic state under controlled seeds;
- buffers, serialization round trip, preprocessing, and inverse transforms.

Exercise the smallest valid case plus meaningful batch, sequence, marks, adjacency,
and optional-input boundaries declared by the model. Choose tolerances from dtype
and operation stability, state them before interpreting results, and save maximum
absolute/relative error per checkpoint with environment facts.

A harness result contains source URL and revision, mapping version, input fixture,
commands, tolerances, output/intermediate/gradient errors, and pass/fail. Shape
agreement, independently initialized outputs, or forward-only closeness is not
parity. On failure, localize the first divergent tensor and return to the port;
do not retain `implementation: upstream` without a passing result.
