# counter

> Counterfactual reasoning for LLM agents. *"What if step N had gone differently?"* — answered structurally, not by guessing.

**Status:** brain dump · v0 design phase · 2026-05-02

---

## The pitch in one paragraph

`counter` is a Python library that ingests an agent's decision trace and lets you ask **causal counterfactual questions over it**: *"if the agent had picked tool B instead of tool A at step 12, what's the predicted distribution over final outcomes?"* It does this by constructing a causal DAG over the trace's decisions, fitting an outcome model on observed (real) trajectories plus replayed mutations, and answering interventional queries. The output is a library you can wrap any agent loop with — usable for (a) auto-generating counterfactual training data for self-improvement loops, (b) attributing failure to specific decisions, (c) driving bandit-style exploration over decision policies.

## Why this exists

### The research gap

LLM agents reason in token space. **They don't natively reason counterfactually**, and prompt-engineering hacks (chain-of-thought, A2P "Abduct-Act-Predict" scaffolding) are the current SOTA. Open problems explicitly named in the 2026 causal-AI literature:

- **Counterfactual generation** that's simultaneously plausible, causally consistent, and actionable.
- **Multi-turn counterfactual inference** — when cause and effect are obscured across many decisions.
- **Distinguishing genuine causal reasoning from pattern recombination** in LLMs.

theCUBE Research called Causal AI Decision Intelligence the *named* 2026 emerging enterprise priority. Anthropic's Sholto Douglas predicted continual learning gets solved this year — counterfactual reasoning is part of how you get there.

### The differentiation

Almost everyone in the agent-infra space is a software engineer who learned LLMs. The combination of **(deep causal-inference background) × (production agentic systems experience)** is genuinely rare. `counter` is the project that converts that combination into a public artifact.

## v0 scope (4–6 weeks of evening/weekend work)

A Python package that:

1. **Ingests an agent trace** in one of: native JSON format, LangSmith export, Langfuse export, OpenTelemetry trace.
2. **Builds a decision DAG** — nodes are decisions (LLM completions, tool selections, memory reads), edges are dependencies (later decisions conditional on earlier ones).
3. **Fits an outcome model** on observed traces — initially logistic regression on engineered features (token count, tool used, model, latency, observed sub-success), later light gradient-boosted models, eventually learned representations.
4. **Answers `do()` queries** in the Pearl sense: `counter.intervene(trace, step=12, action="use_tool_B").outcome_distribution()`.
5. **Ships with one killer demo** — a short notebook where a 200-step agent failure is attributed to a specific decision via counterfactual analysis.

API sketch:

```python
import counter

trace = counter.load("agent-run-2026-04-30.json")
dag = counter.build_dag(trace)

# Counterfactual: what if we'd routed to a smaller model at step 47?
cf = counter.intervene(dag, step=47, action="model=haiku-4.5")
print(cf.outcome_distribution())  # → P(success) = 0.71 ± 0.08

# Failure attribution: which decision most influenced this failure?
attribution = counter.attribute_failure(dag, outcome="failed")
print(attribution.top_k(5))
```

## v1+ stretch goals

- **Synthetic counterfactual training data generation** — sample `do()` queries, run them as actual agent rollouts, build a training set for self-improvement loops.
- **Bandit-driven policy exploration** on top of counter's outcome model — agents that actively explore actions whose counterfactual variance is highest.
- **Integration with `engram`** — use stored memories as prior beliefs for the outcome model.
- **Integration with `replay`** (sibling project) — when counterfactual sampling needs an actual rollout, replay does the deterministic execution.
- **A paper.** Title sketch: *"Counterfactual self-improvement for LLM agents — a structural causal model approach."* Workshop submission target: NeurIPS 2026 or ICLR 2027.

## Technical approach (rough)

Two passes during model fit:

1. **Observational pass.** Use only real traces. Fit `P(outcome | decisions[1..N])` via a structural causal model — ideally a DAG learned from data, fallback to a hand-specified DAG over decision-types.
2. **Interventional pass.** When the user calls `intervene`, we don't just plug values into the observational model — we use the do-calculus to identify the right adjustment set, then either compute analytically (if identifiable) or sample (if not).

For the LLM-completion nodes specifically: we treat them as exogenous random variables but allow conditional sampling on the prompt seen at that step. This is the "causal LLM" abstraction that's unproven but interesting.

Likely dependencies:
- `dowhy` or `causalml` for the do-calculus primitives
- `pydantic` for trace schema
- `polars` for trace data
- `litellm` or direct provider SDKs for LLM-node sampling
- Optional: `pyro` for fully Bayesian outcome models (v1+)

## Why this compounds with the rest

| Compound with | How |
|---|---|
| **engram** | engram stores memories; counter can use them as priors for the outcome model. Memory becomes part of the causal graph. |
| **rigor** (sibling project) | rigor's eval methodology validates counter's interventional predictions. Counter without rigor = vibes; rigor without counter = no causal lens. |
| **Autonomous Insights / ML Research** | These are the loops that produce the traces counter ingests. Counter becomes the introspection layer for them. |
| **trace** | Counter consumes trace-style data. They're complementary: trace = retrieval over knowledge, counter = causal reasoning over decisions. |

## What labs would care about (the elevator pitch)

> *"Most agent self-improvement loops use vibes-based attribution — they retry, observe, hope. We use a structural causal model over the decision trace to answer counterfactual queries directly: which decision most influenced this failure, and what would have happened if it had gone the other way? This turns self-improvement from search into inference."*

That sentence works in a cold DM to anyone at Anthropic / OpenAI / GDM working on agentic post-training, RLAIF, or continual learning.

## Open questions / risks

- **Identifiability** — for many counterfactual queries over LLM decisions, the back-door criterion isn't satisfied. We may need to fall back to bounded sampling or assume strong-ignorability. This is the technical risk that makes the paper interesting (or, alternatively, makes it not work).
- **DAG specification** — learning the DAG from traces is hard with limited data. The pragmatic v0 hand-specifies it for common decision types.
- **LLM-as-exogenous** — treating LLM completions as exogenous random variables is a modeling choice; some failure modes are *internal* to the LLM and our framing doesn't capture them.

## Decision log

- 2026-05-02 — repo created. Brain dump v0 from chat with the wiki agent.

## Cross-references

- Brain wiki: [career plan 2026–2028](internal-brain-wiki-career-plan) — counter is move #2/#3 of Track B.
- Brain wiki: [Senior DS career reference](internal-brain-wiki-career-reference) — research-portfolio differentiation rationale.
- Sibling repo: `../rigor` — eval harness with experimental rigor.
- Sibling repos: `related local agent-infra repos` — existing pieces of the agent-infra portfolio.
- Reference paper: [Abduct-Act-Predict scaffolding for causal inference (Sept 2025)](https://arxiv.org/pdf/2509.10401) — current SOTA in agent counterfactuals.
