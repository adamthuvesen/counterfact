# Causal And Statistical Invariants

These are the honesty guarantees `counterfact` is built to defend. A change that
weakens any of them is a regression even if the tests pass.

- `CausalEstimate.identifiability` must be one of `identified`, `bounded`, or `unidentified`, and the rest of the object must make that label defensible.
- Single-class real corpora are not model-fit inputs. Surface the degenerate case as `unidentified` with a concrete `NextStep`.
- Prediction uncertainty and identifiability uncertainty are different. Do not blur bootstrap CIs, sensitivity bounds, support gaps, and replay requirements.
- If a query needs a prompt rewrite, hidden state change, or unavailable arm, return an honest replay/support next step rather than an estimated effect.
- `pass_rate_by_arm()` is descriptive, not causal. Keep that distinction visible in demos, docs, tests, and CLI output.
- Synthetic SCM tests should remain deterministic by seed and recover the known headline effect within the acceptance tolerance.
