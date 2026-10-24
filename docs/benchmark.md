# Estimator Benchmark

The original synthetic SCM in `bench/synthetic/scm.py` is a main-effects logistic model, the same family as the outcome model `counterfact` fits. Recovering its effect shows that the pipeline works. It does not show that the engine is honest when its assumptions fail. This benchmark is built to fail estimators.

Every scenario asks one question: the effect of `model_choice = sonnet` against `haiku` on task success. Each scenario breaks one assumption. All variables are discrete, so the true effect is an exact sum over the model, not a simulation. `tests/unit/test_benchmark_scenarios.py` checks each true effect against a simulated intervention.

The table compares three estimators on the same corpora:

- **Naive:** the pass-rate difference between arms (`pass_rate_by_arm`).
- **Adjust-all:** the engine before data-checked identification. It fits one logistic model on every logged decision, including decisions after the treatment, then applies the g-formula. It always returns a number.
- **Engine:** `counterfact.intervene`. It returns a label. It returns a cross-fitted AIPW contrast with a 95% CI only when the corpus identifies the contrast, and Manski bounds when it does not.

## Results

Full table, 500 seeds per scenario: [benchmark-results.md](benchmark-results.md). Raw summaries: [benchmark-results.json](benchmark-results.json).

```bash
uv run counterfact bench suite --seeds 500 --markdown docs/benchmark-results.md --json docs/benchmark-results.json
```

The run takes about 40 seconds on a laptop and makes no API calls.

## What the numbers show

- **The engine labels every scenario correctly.** Across 4,000 corpora it returns `identified` in all six scenarios where the data identify the effect, and `bounded` in both where they do not. The value of `bounded` is the refusal, not the interval: Manski bounds on a binary outcome assume nothing, so they span about [-0.31, +0.69] here. That includes zero, so they cannot say which model is better. They say how much an assumption would have to carry.
- **The intervals are calibrated.** When the engine reports `identified`, its 95% CI covers the true effect in 94% to 96% of seeds. That includes the n=80 case. With 500 seeds the Monte Carlo standard error of each coverage figure is about 1 percentage point.
- **Logged propensities beat a hidden confounder.** In `unlogged_confounder_logged_propensity`, hidden task difficulty drives both the model choice and the outcome. Naive and adjust-all are both biased by about +0.16. The engine's bias is +0.002, because the logged propensities are the true assignment probabilities and inverse weighting does not need the confounder.
- **Remove the propensities and the engine refuses.** `unlogged_confounder` is the same process with the propensities removed. Naive and adjust-all still report a gap that is +0.16 too high. The engine returns `bounded`.
- **Adjusting for a post-treatment decision is a real failure.** In `post_treatment_retry`, the agent retries after a failed first attempt, which the model choice caused. Adjust-all conditions on the retry and is biased by -0.091. The engine adjusts only for earlier steps and is unbiased.
- **The naive table can get the sign wrong.** In `interaction`, sonnet helps only after the agent runs the tests, and it is mostly routed to those runs. The true effect is -0.042. The naive gap is about +0.27.

## Where the baselines held up

Two results are less dramatic than the scenario names suggest. They stay in the table because a benchmark that hides them would be a worse benchmark.

- **Interaction did not break the main-effects g-formula.** Even with a strong tool × model interaction, adjust-all stayed within +0.009 of the truth. Standardizing over the observed covariate distribution absorbed most of the misspecification here. This scenario separates naive from causal estimators. It does not separate AIPW from the g-formula.
- **Adjust-all was close under the positivity violation.** In `positivity_violation`, adjust-all is off by only +0.011, because its main-effects model happens to be the true model and extrapolates correctly to arms the data never show. Nothing in the data proves that. The engine returns `bounded` because a correct guess is still a guess.

## Policy evaluation: decisions that react to the run

Agents do not make one decision. A retry depends on whether the tests passed, and the tests passing depends on the model the agent chose earlier. `bench/synthetic/sequential.py` builds that loop: a randomized model choice, a test run whose result is logged, and a retry that the logging policy takes more often after a failure. Hidden task difficulty affects both attempts. The test result is a time-varying confounder: the earlier decision causes it, and it drives the later decision and the outcome.

`counterfact.evaluate_policy` answers the question an agent builder actually asks: how would success change if the agent followed a policy, such as "always use sonnet and retry only when the tests fail"? It estimates V(π) - V(logged) with sequential doubly robust estimation. The table in [benchmark-results.md](benchmark-results.md#policy-evaluation) scores it on 500 seeds. Each seed draws one corpus of 2,000 runs, and the five identified policy cases are evaluated on the same corpora.

- **Filtering the logs gets both retry policies backwards.** Averaging the runs that already followed a policy says never retrying raises success by 0.145 and always retrying lowers it by 0.154. The truths are -0.093 and +0.040. Runs that skipped the retry mostly passed first time, so they look good. That is the time-varying confounder at work.
- **Sequential DR recovers every policy value.** Its bias is at most 0.0014 and its 95% CIs cover the truth in 94% to 97% of seeds, including a policy that sets both the model and the retry rule.
- **Double robustness shows up where it should.** When the outcome models cannot see `tests_passed`, the ICE g-formula is off by -0.177. DR stays unbiased, because the logged propensities are exact. Giving the outcome models the confounder does not change DR's bias, but it cuts its RMSE from 0.018 to 0.003.
- **No propensity, no point estimate.** With the retry propensity removed from the log, the engine returns `bounded` on every seed.

`tests/unit/test_policy_evaluation.py` also checks that, with a single controlled decision, the policy contrast agrees with the single-decision AIPW contrast within 0.02.

## Limits

- Every scenario is synthetic and discrete, with one binary outcome.
- Policies are deterministic rules over one history feature per decision type. Stochastic policies and rules that depend on several features are not supported.
- The sequential scenario has two controlled decisions. Trajectory weights grow with the number of controlled decisions, so longer horizons will need more runs; the reported effective sample size shows when that bites.
- The suite measures the engine against processes it can describe. It does not show how often real agent traces satisfy the logging the engine needs. The real-agent corpora under `bench/real/` are the evidence for that.
