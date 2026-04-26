# RFC 0005 — Feedback-Loop Simulation

| Field | Value |
|---|---|
| Status | Implemented (1.1.0-dev) |
| Author | Michael Kupermann |
| Date | 2026-04-26 |
| Reviewers | (open for comment) |

## Context

`equiml audit`, `equiml verify`, and `equiml monitor` evaluate a model against a static or streaming set of decisions. None of them ask the question that operations engineers, model-risk reviewers, and regulators all eventually ask: *what does this model do to the world it operates in?*

The textbook failure mode is well documented and almost unaddressed in fairness tooling:

- A lending model is deployed. It rejects more applicants from group B than from group A.
- Rejected applicants do not produce repayment data; their outcomes are never observed.
- The next training set is dominated by group A's repayment history.
- The next model is more confident about group A's outcomes and less informed about group B's.
- The selection gap widens.

Ensign et al. (*Runaway Feedback Loops in Predictive Policing*, FAT\* 2018) established this as a systemic property of any system where the model's decisions affect what data the system collects next. The same loop has been documented in hiring (Kleinberg & Mullainathan, 2019), credit decisioning, child-welfare risk prediction, and university admissions. A point-in-time audit cannot see this loop. A streaming monitor can only see it after it has happened.

EquiML 1.1 already ships drift detection (RFC 0003), which observes deployment-time drift after the fact. RFC 0005 adds a tool that simulates the loop *before* deployment, so a team can answer "if we ship this model with this decision rule and this feedback regime, what does the fairness gap look like in five retrain cycles?" without running the experiment on real applicants.

## Decision

EquiML 1.1 ships `equiml/simulation.py` containing:

1. A `DecisionRule` enum with `THRESHOLD` (proba > t) and `TOP_K` (the highest K scores). The decision rule is what the deployed system actually does with the score.
2. A `FeedbackRule` enum with `SELECTION_BIAS` (only selected applicants' outcomes enter the next training set) and `PERFECT_INFO` (all outcomes observed; the no-feedback baseline). `LABEL_FLIP` is reserved for a richer simulation where successful selected applicants retroactively get positive labels.
3. A `RoundResult` dataclass capturing one round's metrics, training-pool size, selection count, and per-group selection rates.
4. A `FeedbackLoopSimulator` class that takes a `model_factory` callable, runs K rounds, and returns the per-round trajectory. Each round trains a fresh model — we do not reuse a fitted estimator across rounds, because a real retrain cycle does not.
5. A `summary()` static method that produces the headline number: mean / stddev DP and EO across rounds, accuracy mean, the linear-fit slope of DP over rounds, and a one-sentence drift headline ("DP gap moved from 0.04 at round 0 to 0.18 at round 9").

The split between `DecisionRule` and `FeedbackRule` is deliberate: a team can hold the decision policy fixed and vary the assumed information-recovery regime, or vice versa. A simulator that conflates the two would forbid the most useful counterfactuals.

## API

```python
from equiml.model import Model
from equiml.simulation import (
    DecisionRule,
    FeedbackLoopSimulator,
    FeedbackRule,
)

def make_model():
    # Fresh model each round — do NOT pass a fitted estimator.
    return Model(algorithm="logistic_regression")

sim = FeedbackLoopSimulator(
    model_factory=make_model,
    sensitive_features=["group"],
    decision_rule=DecisionRule.THRESHOLD,
    feedback_rule=FeedbackRule.SELECTION_BIAS,
    n_rounds=10,
    random_state=42,
    threshold=0.7,
)

results = sim.run(
    X_train_init=Xtr, y_train_init=ytr, sensitive_train_init=str_,
    X_test=Xte, y_test=yte, sensitive_test=ste,
)

summary = FeedbackLoopSimulator.summary(results)
# summary["dp_first"], summary["dp_last"], summary["dp_slope"],
# summary["drift_headline"]
```

Each round produces a `RoundResult` with:

```python
@dataclass
class RoundResult:
    round_idx: int
    metrics: dict[str, float]                       # DP, EO, accuracy
    per_sensitive: dict[str, dict[str, float]]      # mirrors audit JSON shape
    n_train: int                                    # training pool size after this round's feedback
    n_selected: int                                 # decisions made this round
    selection_rate_per_group: dict[str, float]      # per-group selection rate
```

`per_sensitive` mirrors the audit JSON's per-feature block so downstream tools (model card, drift report) can render simulator output without a new template.

## Modelling assumptions

A simulation only answers the question its assumptions allow. The simulator's assumptions, in order of how much they bend reality:

- **The applicant pool is sampled from the initial training distribution.** Real applicant pools shift independently — economic conditions change, marketing changes who applies, the population ages. The simulator does not model that. This is the single largest gap between simulation and reality.
- **The held-out test set is fixed across rounds.** That isolates the model-drift signal cleanly. In production the test distribution drifts too, and the monitor in RFC 0003 is the appropriate tool for that.
- **Feedback rules are deterministic.** `SELECTION_BIAS` always drops every rejected applicant's outcome. Real systems sometimes recover noisy signal on rejected applicants (a hand-reviewed sample, a downstream measurement, a competitor's data). Partial-information feedback is future work.
- **No strategic response.** Applicants do not change their behaviour in response to the model's decisions. The literature on strategic classification (Hardt et al., 2016) shows this assumption matters when the modelled subjects are sophisticated; the simulator does not capture it.

These are documented in the module docstring and surface in the RFC because the value of the simulation depends on the reader knowing them.

## CLI integration

This RFC proposes — but does not implement — a new `equiml simulate` subcommand. The implementation lands in a synthesis pass that touches `equiml/cli.py`.

```bash
equiml simulate data.csv \
    --target outcome \
    --sensitive group \
    --rounds 10 \
    --decision-rule threshold \
    --threshold 0.7 \
    --feedback-rule selection-bias \
    --algorithm logistic_regression \
    --output simulation.json
```

The output JSON shape mirrors the per-round dataclass, plus the summary block:

```json
{
  "_meta": {
    "equiml_version": "1.1.0",
    "decision_rule": "threshold",
    "feedback_rule": "selection-bias",
    "n_rounds": 10,
    "random_seed": 42
  },
  "rounds": [
    {"round_idx": 0, "metrics": {...}, "n_train": 480, ...},
    ...
  ],
  "summary": {
    "dp_first": 0.04, "dp_last": 0.21, "dp_slope": 0.018,
    "drift_headline": "DP gap moved from 0.04 at round 0 to 0.21 at round 9 (10 rounds, slope +0.0180/round)"
  }
}
```

Exit codes follow the existing convention:

| Code | Meaning |
|---|---|
| 0 | Success — simulation completed. |
| 2 | Data error (file not found, columns missing). |
| 6 | Simulation drift threshold breached (when `--max-dp-slope` is provided and exceeded). Reserved; not implemented in v1. |

## Alternatives considered

### Full agent-based simulation

Pro: captures applicant-pool drift, strategic response, and macroeconomic shocks.
Con: a 10x larger module, dramatically more assumptions, and an audience overlap of approximately zero with EquiML's current users. An agent-based fairness simulator is a separate project; if one of our users wants one, they will use Mesa or AnyLogic, not a CLI auditor.

**Verdict:** out of scope. We ship the smallest tool that captures the canonical feedback loop honestly.

### Pyro / probabilistic-programming foundation

Pro: would make the modelling assumptions formal and let users substitute their own priors.
Con: a heavy dependency, an unfamiliar interface for the platform engineers and analysts who use EquiML, and a poor fit for the deterministic-feedback abstraction we want.

**Verdict:** plain numpy. The simulator's assumptions live in the docstring and the RFC, where they can be audited.

### `gym`-style API

Pro: aligns with reinforcement-learning conventions.
Con: a `step()` / `reset()` / `done` interface buys nothing here. The simulator runs end-to-end; no agent is choosing actions during the loop.

**Verdict:** plain `run()` returning a list. Easier to read, easier to test.

### Reuse `BiasMonitor` / `FairnessDriftMonitor`

Pro: one less class.
Con: those tools observe; the simulator generates. Conflating them confuses what is real (monitor) and what is hypothetical (simulator). The simulator's output can feed a monitor for downstream rendering, but the loop logic is its own concern.

**Verdict:** new module. The simulator's per-round output uses the same `metrics` / `per_sensitive` shape as audit JSON so existing renderers can pick it up.

## Consequences

### What gets easier

- A team can quantify deployment risk before deployment. "If we ship this threshold with this feedback regime, our DP gap grows at +0.018 per retrain" is a concrete claim that goes into a model card or a model-risk review.
- Procurement and compliance reviewers get a textbook-failure-mode trajectory chart from the same artefact pipeline as the audit JSON. The argument shifts from "we audited the model" to "we audited the model and simulated five retrain cycles of the most likely failure regime."
- A red-team workflow opens up: vary the decision rule, vary the feedback rule, see which combinations produce the steepest DP slope. That output is more actionable than a single audit number.

### What gets harder

- Simulation results carry assumptions. A team that publishes a simulation without naming the feedback rule is publishing a number whose meaning is unclear. We mitigate by recording the rule in the result JSON's `_meta` and emitting it in any rendered report.
- The temptation to over-trust a simulation is real. We address this in the RFC, in the module docstring, and in the example. A future expansion should include an "assumption summary" block in the output that the model-card generator surfaces verbatim.

### Backward compatibility

Net new module. No existing CLI surface, API, or output shape changes. The simulator is opt-in.

## Future work (not in this RFC)

- **Applicant-pool drift.** Let the next round's applicant pool be sampled from a distribution that itself moves — e.g. group composition shifts by a known rate per round. Currently the pool is bootstrapped from the initial training set.
- **Strategic response.** A simple parametric model where applicants who fall just below the threshold can spend effort to nudge a feature. Compatible with the existing decision rule abstraction.
- **Intersectional simulation.** Today the simulator tracks one primary sensitive feature for the metrics. A second sensitive feature could enter as a secondary dimension on `per_sensitive` and `selection_rate_per_group`.
- **Partial-information feedback.** A `FeedbackRule.NOISY_REJECTION` that recovers a stochastic fraction of rejected applicants' outcomes — closer to systems that hand-review a sample of rejections.
- **Counterfactual simulation.** What would the trajectory have looked like if the policy had been DP-constrained from round 0? The simulator can run two parallel chains; the comparison is the deliverable.
- **Simulation-aware policy.** A `simulation:` block in `fairness.yaml` that gates on simulator output (e.g. "DP slope must be < 0.01 over 10 rounds with selection-bias feedback") and runs as a CI gate.

## Honest limitations

The simulator answers a specific question: under a stated decision rule and a stated feedback rule, how does the per-round fairness metric behave? It does not predict deployment outcomes. It does not capture applicant-pool dynamics, macroeconomic shocks, strategic response, regulatory change, or the dozens of other forces that move a real production system. A model that looks fair under simulation can still fail in deployment; a model that looks unfair under simulation will likely fail worse. The simulator is a stress test, not a forecast.

The RFC is unambiguous about this because the alternative — letting a team publish a simulation chart as a deployment safety claim — is the failure mode this RFC was written to prevent.
