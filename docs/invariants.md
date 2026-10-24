# Causal and Statistical Invariants

These are the honesty guarantees `counterfact` defends. A change that weakens any of them is a regression even if the tests pass.

- `CausalEstimate.identifiability` must be one of `identified`, `bounded`, or `unidentified`, and the rest of the object must make that label defensible.
- Single-class real corpora are not model-fit inputs. Surface the degenerate case as `unidentified` with a concrete `NextStep`.
- Prediction uncertainty and identifiability uncertainty are different. Do not blur sampling CIs, identification bounds, support gaps, and replay requirements.
- `identified` requires logged randomization that the data confirm: a propensity on every analyzed run and positive probability for both contrasted arms on every run, shown by the log rather than inferred from `valid_actions`. A taxonomy stance never licenses a point estimate by itself.
- Adjustment sets hold only decisions logged in steps strictly before the treatment. Never condition on a post-treatment or same-step decision.
- `outcome_delta` is always a contrast against a named baseline arm, never a level.
- If a query needs a prompt rewrite, hidden state change, or unavailable arm, return an honest replay/support next step rather than an estimated effect.
- `pass_rate_by_arm()` is descriptive, not causal. Keep that distinction visible in demos, docs, tests, and CLI output.
- Synthetic SCM tests should remain deterministic by seed and recover the known headline effect within the acceptance tolerance.
- Policy evaluation follows the same rules at every controlled decision: logged propensities, positive probability for the policy's action, and history from strictly earlier steps only. The matched-runs average stays labeled descriptive.
