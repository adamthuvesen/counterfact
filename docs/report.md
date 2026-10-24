# Honest Counterfactuals for Agent Traces

Technical report for `counterfact` 0.1.0. Every benchmark number below comes from `docs/benchmark-results.json` and can be reproduced with one command in about two minutes on a 10-core laptop.

## Summary

Teams that build LLM agents ask counterfactual questions of their logs. Would this run have passed with the larger model? Should the agent retry less? The usual answer is a pass-rate table split by the decision. That table is confounded: agents pick the larger model for harder tasks and retry after failures, so the runs that took an action differ from the runs that did not.

`counterfact` answers these questions from logged traces with an identification check first and an estimate second. Every answer carries one of three labels, and the label is checked against the data:

- `identified`: the log records the assignment probability of every analyzed decision, and both arms have positive probability on every run. The engine returns a doubly robust contrast with a 95% CI.
- `bounded`: both arms occur, but the log does not prove how they were assigned. The engine returns Manski bounds, which assume nothing, or no interval when the data cannot support one.
- `unidentified`: the corpus cannot answer the question. The engine returns a concrete next step, such as the runs to collect.

The engine is scored on structural causal models with exact true effects, built so that each one breaks one assumption. Over 4,000 single-decision corpora it labels every scenario correctly, and its 95% CIs cover the truth 94% to 96% of the time. For whole policies over a sequential agent, the common practice of averaging the runs that already followed the policy gets both retry policies backwards. The sequential doubly robust estimate has a mean bias of at most 0.0014.

![Single-decision bias by estimator](figures/single-decision-bias.svg)

## 1. Problem

A trace is a sequence of steps. Each step holds decisions (a model call, a tool call, a retry, a stop) and observations (tool output, test results). The run ends with an outcome, such as whether the tests pass.

The question for one decision is a contrast between two interventions:

$$\Delta = E[Y \mid do(A = a)] - E[Y \mid do(A = b)]$$

Here $A$ is the first occurrence of a decision type in the run, $a$ is the target action, and $b$ is a named baseline. The default baseline is the action the focal trace actually chose.

The question for a policy $\pi$ is the change in success if the agent followed $\pi$ at every decision it controls, and acted as logged everywhere else:

$$\Delta_\pi = V(\pi) - V(\text{logged})$$

Two features of agent logs make these questions hard:

- **Upstream confounding.** Routers send hard tasks to strong models. Difficulty is rarely logged.
- **Time-varying confounding.** The model choice changes whether the first test run passes, and the test result drives the retry. Adjusting for the test result to study the retry blocks part of the model's effect. Ignoring it confounds the retry.

## 2. Identification as a data check

A point estimate is only as good as its identification argument. A decision-type taxonomy routes each question first. Questions that need replay, such as a prompt rewrite, go straight to `unidentified`. Memory-content questions need a back-door argument the log cannot check, so they go to `bounded` with no interval. A taxonomy entry can only send a question toward a refusal. It never grants `identified`. For model, tool, temperature, and retry decisions, the engine checks the log.

A decision supports `identified` when all of these hold:

1. The decision logs its propensity, the probability the agent's policy gave the chosen action.
2. The log shows positive probability for both contrasted arms on every analyzed run. The chosen arm proves its own mass. With two valid actions, a propensity below 1 proves mass on the other one. A named randomization policy (`uniform`, `epsilon_greedy`) counts only when the logged propensity equals the value that policy implies.
3. The adjustment set holds only decisions from steps strictly before the treatment. A same-step or later decision is never a covariate.

When a decision fails rule 1 or 2 but both arms occur, the label is `bounded`. When the corpus has one outcome class, no support for an arm, or a question that needs replay (such as a prompt rewrite), the label is `unidentified`.

Policy evaluation applies the same rules at every decision the policy controls. Positivity is checked only on the reachable part of each run: the decisions up to and including the first one where the logged action leaves the policy.

## 3. Estimators

### One decision

The engine fits outcome models with 5-fold cross-fitting and forms the augmented inverse propensity weighted (AIPW) score for each run $i$:

$$\psi_i(a) = \hat\mu_a(X_i) + \frac{1\{A_i = a\}}{e_i}\,(Y_i - \hat\mu_a(X_i))$$

Here $e_i$ is the logged propensity of the chosen action and $\hat\mu_a$ is the held-out outcome model. The contrast is the mean of $\psi_i(a) - \psi_i(b)$, and the 95% CI uses the sample variance of those scores. The logged propensity is the true assignment probability, so the estimate stays unbiased even when the outcome model is wrong or a confounder is missing from $X$.

Two cross-checks are reported next to it: the g-formula with the same outcome model, and the Hajek IPW estimate. A large gap between them is a diagnostic, not a correction.

When a model, tool, temperature, or retry question is `bounded`, the engine returns Manski bounds for a binary outcome. They assume nothing about the unobserved arm, so they are wide by construction. Their value is the refusal: they show how much an unstated assumption would have to carry.

### A policy over several decisions

For a deterministic policy $\pi$ over controlled decisions $k = 1, \dots, K$, the engine uses sequential doubly robust estimation (iterated conditional expectations with importance-weighted residuals). It fits $\hat Q_K(H_K, A_K) \approx E[Y \mid H_K, A_K]$, then works backward:

$$\hat Q_k(H_k, A_k) \approx E[\hat Q_{k+1}(H_{k+1}, \pi(H_{k+1})) \mid H_k, A_k]$$

The per-run score is

$$\hat Q_1(H_1, \pi(H_1)) + \sum_{k=1}^{K} W_k \left(\hat Q_{k+1}(H_{k+1}, \pi(H_{k+1})) - \hat Q_k(H_k, A_k)\right), \quad W_k = \prod_{j \le k} \frac{1\{A_j = \pi(H_j)\}}{e_j}$$

with $\hat Q_{K+1} = Y$. The sum stops at the first decision where the run leaves the policy. The fits are cross-fitted. A held-out decision deeper than any training run falls back to the training mean, so a held-out outcome never enters its own prediction.

The engine reports two cross-checks: the plug-in ICE value and self-normalized importance sampling (SNIS). It also reports the effective sample size of the weights and the descriptive average over runs that already matched the policy.

## 4. Benchmark design

The original synthetic model in `bench/synthetic/scm.py` is a main-effects logistic model, the same family the outcome model fits. Recovering its effect shows that the code runs. It does not show that the engine is honest when its assumptions fail.

The benchmark uses discrete structural causal models, so each true effect is an exact sum over the model, not a simulation. A unit test checks each truth against a simulated intervention. Each single-decision scenario breaks one assumption:

| Scenario | What it breaks |
| --- | --- |
| `randomized` | Nothing. The control case. |
| `small_n` | Sample size: n = 80. |
| `logged_confounding` | Naive pass rates. The arm depends on a logged tool choice. |
| `interaction` | Outcome-model form. A strong tool × model interaction. |
| `unlogged_confounder_logged_propensity` | Covariate adjustment. Hidden difficulty drives the arm, but the propensity is logged. |
| `unlogged_confounder` | Point identification. The same process with no propensity. |
| `post_treatment_retry` | Adjust-everything. A retry that reacts to the treatment. |
| `positivity_violation` | Positivity. The model is routed deterministically by tool. |

Each scenario compares three estimators on the same corpora: the naive pass-rate gap, an adjust-everything g-formula (the engine before identification checks), and the engine.

The sequential scenario adds a time-varying confounder. A randomized model choice affects whether the first test run passes. The logging policy retries 70% of the time after a failure and 20% after a pass. Hidden difficulty affects both attempts.

## 5. Results

500 seeds per scenario. Bias is the mean error against the exact truth. Engine bias and coverage use only seeds labeled `identified`. The full tables are in [benchmark-results.md](benchmark-results.md).

| Scenario | Truth | Naive bias | Adjust-all bias | Engine label | Engine bias | 95% CI coverage |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `randomized` | +0.234 | +0.002 | +0.000 | 100% identified | +0.002 | 95% |
| `small_n` | +0.234 | -0.011 | -0.033 | 100% identified | -0.009 | 94% |
| `logged_confounding` | +0.234 | +0.106 | +0.001 | 100% identified | +0.002 | 94% |
| `interaction` | -0.042 | +0.310 | +0.009 | 100% identified | +0.001 | 96% |
| `unlogged_confounder_logged_propensity` | +0.217 | +0.160 | +0.157 | 100% identified | +0.002 | 94% |
| `unlogged_confounder` | +0.217 | +0.159 | +0.156 | 100% bounded | | |
| `post_treatment_retry` | +0.199 | +0.001 | -0.091 | 100% identified | +0.001 | 95% |
| `positivity_violation` | +0.234 | +0.192 | +0.011 | 100% bounded | | |

Four findings stand out:

- **The naive table can get the sign wrong.** In `interaction` the true effect is -0.042 and the naive gap is about +0.27.
- **Logged propensities beat a hidden confounder.** With difficulty hidden, naive and adjust-all are both off by about +0.16. The engine is off by +0.002, because inverse weighting by the true assignment probability does not need the confounder.
- **Remove the propensity and the engine refuses.** The same process without the propensity returns `bounded` on every seed. The other two estimators still report a gap that is +0.16 too high.
- **Adjusting for everything is a real failure.** Conditioning on a retry that the treatment caused biases adjust-all by -0.091. The engine adjusts only for earlier steps.

Two baselines held up better than the scenario names suggest, and the table keeps them. The main-effects g-formula absorbed the interaction. Under the positivity violation it extrapolated correctly because its model happened to be the true model. Nothing in the data proves that, so the engine returns `bounded`.

### Policies

![Policy evaluation bias by estimator](figures/policy-bias.svg)

2,000 runs per corpus, 500 seeds.

| Policy | Truth | Matched-runs bias | ICE bias | DR bias (RMSE) | 95% CI coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| always retry, outcome model blind to `tests_passed` | +0.040 | -0.193 | -0.177 | +0.000 (0.018) | 94% |
| always retry | +0.040 | -0.193 | -0.001 | +0.000 (0.003) | 94% |
| never retry | -0.093 | +0.238 | +0.008 | +0.000 (0.006) | 96% |
| retry on fail | +0.040 | +0.025 | -0.004 | +0.000 (0.003) | 94% |
| sonnet, then retry on fail | +0.172 | +0.024 | -0.005 | +0.001 (0.012) | 97% |
| retry on fail, retry propensity not logged | +0.040 | | | 100% bounded | |

- **Filtering the logs reverses both retry policies.** The matched-runs average says never retrying raises success by 0.145 and always retrying lowers it by 0.154. The truths are -0.093 and +0.040. Runs that skipped the retry mostly passed first time.
- **Double robustness shows up where the theory says it should.** When the outcome models cannot see `tests_passed`, ICE is off by -0.177 and DR stays unbiased. Giving the models the confounder cuts DR's RMSE from 0.018 to 0.003.
- **The two estimators agree where they should.** With one controlled decision, the policy contrast matches the single-decision AIPW contrast within 0.02 (`tests/unit/test_policy_evaluation.py`).

## 6. Real-agent ground truth

Synthetic models test the engine against processes it can describe. A real agent is the harder test. The harness in `bench/real/` runs a coding agent on fixed fixtures with a routed-trial design:

- Each run is randomized into one of two arms. In the trial arm, the model is a 50/50 coin flip. In the router arm, a fixture-specific router sends hard fixtures to the large model, as a production router would.
- The trial arm gives an unbiased ground-truth effect from one corpus. The router arm gives a confounded log of the same agent on the same fixtures.
- The analysis compares the naive gap and the engine's AIPW estimate on the routed runs against the trial truth. It then removes the logged propensities from the routed runs and checks that the engine refuses.

The harness and the analysis are merged and tested. The paid run (800 traces, a cap of $18) has not been run yet. This report will add its results when it is.

## 7. Limits

- Every benchmark scenario is synthetic and discrete, with one binary outcome.
- `identified` needs logged propensities. Most production agents do not log them today. The engine's answer for those logs is `bounded`, which is correct and not very useful. Logging propensities, or randomizing a small share of decisions, is the practical fix.
- Policies are deterministic rules over one history feature per decision type. Stochastic policies and rules over several features are not supported.
- Trajectory weights grow with the number of controlled decisions. Long horizons need more runs. The reported effective sample size shows when that limit applies.
- Manski bounds are honest but wide. Sensitivity analysis that trades a stated assumption for a tighter interval is future work.

## Reproduce

```bash
uv run counterfact bench suite --seeds 500 --markdown docs/benchmark-results.md --json docs/benchmark-results.json
uv run python scripts/build_figures.py
```

The suite makes no API calls. Design notes for each scenario are in [benchmark.md](benchmark.md). The identification guarantees are in [invariants.md](invariants.md).
