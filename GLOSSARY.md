# counterfact

counterfact answers causal questions about agent traces, and says when the logged data cannot support an answer.

## Language

**Run**:
One agent trace of one task: steps, decisions, observations, and one outcome (`schema/models.py::Run`).
_Elsewhere_: an eval-audit run is one agent over the whole task set, and a heist run is one invocation over agents x tasks. `export-runs` writes one eval-audit `RunRecord` per counterfact run.

**Corpus**:
A directory of run JSON files, read through `corpus_io.load_corpus`.

**Decision**:
One choice the agent made at one step, such as a `model_call` or a `retry`, with its chosen action and any logged randomization.

**Arm**:
One chosen action of a decision type, such as `sonnet` for `model_call` (`baselines.py::pass_rate_by_arm`).
_Elsewhere_: in eval-audit, an arm is the treatment side or the control side of a claim.

**Propensity**:
The logged probability that the policy chose the logged action. Identification needs it.

**Outcome**:
The verifier's result for a run: `binary`, `categorical`, or `continuous`. Only binary outcomes fit today, and only binary runs export to eval-audit.

**Identifiability**:
The label on every causal answer: `identified`, `bounded`, or `unidentified`. `identified` needs logged propensities and checked support.

**Focal intervention**:
An `intervene` query whose decision type and baseline come from one decision in a focal run. The answer is still a population contrast over the corpus, not an effect for that run.
It differs from a first-occurrence `contrast` and from a sequential policy estimand.

**Scenario**:
A synthetic structural causal model in the benchmark suite that breaks one assumption a naive analysis needs (`bench/synthetic/scenarios.py`).
_Elsewhere_: an eval-audit sensitivity scenario is a re-analysis with one setting changed.

**Harness**:
The system that produced a trace. The eval-audit export takes it from `metadata.extra.harness`, else `counterfact-trace`.
_Elsewhere_: heist is a harness and exports `"heist"`.
