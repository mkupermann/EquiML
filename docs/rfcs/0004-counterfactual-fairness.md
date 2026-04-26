# RFC 0004 — Naive Counterfactual Fairness + Proxy Detection

| Field | Value |
|---|---|
| Status | Implemented (1.1.0-dev) |
| Author | Michael Kupermann |
| Date | 2026-04-26 |
| Reviewers | (open for comment) |

## Context

`equiml audit` answers a group-fairness question: "do the model's
selection rates differ across groups, and by how much?" That is the
right question for a regulator who has to compare cohorts. It is not the
right question for the engineer who has to defend a single decision.

A loan officer reviewing a denied application does not want a
demographic-parity number. They want to know: *would the model have
decided differently for this person if their gender had been recorded
differently, holding the rest of the file constant?* That is the
counterfactual fairness question (Pearl 2009; Kusner et al. 2017).

A real causal counterfactual requires a structural causal model: a graph
declaring which features cause which, where the protected attribute sits
in the graph, and what `do(gender = ¬gender)` propagates to downstream
features. EquiML does not ship that. Spec'ing the graph correctly is
domain work the audit tool cannot do for the team using it; getting it
wrong gives the most dangerous kind of false confidence — *causally
labelled* false confidence.

What the audit tool *can* do without a graph is the **naive
counterfactual**: flip the protected attribute on the test set, hold
everything else constant, re-predict, and report how often the
prediction changes. That number is a sensitivity. It is not a causal
claim. But it tells the engineer something the demographic-parity
metric does not: how much of the model's behaviour depends on the
protected attribute as a feature.

The companion question — *which non-protected features carry the
protected signal?* — is what this RFC calls **proxy detection**.
"Salary history correlates with gender" is not a causal claim either,
but a model that loses its protected-attribute sensitivity when salary
history is neutralised is leaning on salary history as a stand-in.
Surfacing that ranking is the load-bearing piece of this feature.

## Decision

EquiML 1.1 ships a new module `equiml/counterfactual.py` exporting:

1. **`CounterfactualResult`** — a dataclass with
   - `flip_rate: float` — share of test-set predictions that change
     when the protected attribute is flipped (binary) or cycled and
     averaged (multi-class),
   - `mean_prediction_shift: float | None` — for models with
     `predict_proba`, the mean of `|p_flipped − p_original|` for the
     positive class; `None` for the fairness-mitigated path that has no
     well-defined probability output,
   - `proxy_features: list[ProxyFeature]` — empty by default, populated
     by the proxy ranker,
   - `n_samples: int`,
   - `notes: list[str]` — caveats raised during the run (e.g.
     multi-class cycling, missing `predict_proba`).

2. **`ProxyFeature`** — a dataclass with
   - `feature_name: str`,
   - `flip_rate_with: float` — flip rate on the original `X_test`,
   - `flip_rate_without: float` — flip rate on a copy where the
     candidate feature has been set to its global neutral value (mean
     for numerical, mode for categorical),
   - `proxy_strength: float` — `flip_rate_without − flip_rate_with`.
     Positive ⇒ neutralising the candidate raised the protected-
     attribute flip rate, i.e. the candidate was carrying / masking the
     protected signal. Larger ⇒ more proxy-like.

3. **`compute_counterfactual_audit(model, X_test, sensitive_feature, ...)`**.
   `sensitive_feature` may be a column name in `X_test` or a standalone
   `pd.Series`. Binary protected attributes get a single swap; multi-
   class attributes get a cycle (one frame per non-original level), and
   the result averages across cycles.

4. **`compute_proxy_features(model, X_test, sensitive_feature,
   candidate_features=None, top_k=10)`**. For each candidate, sets the
   candidate to its global neutral value and re-runs the counterfactual
   audit. Returns the list sorted by descending `proxy_strength`. The
   protected column is filtered out automatically even if the caller
   passes it in `candidate_features`.

The module-level docstring states the limitation explicitly:

> This is sensitivity analysis, not causal counterfactual fairness in
> the Pearl/Kusner sense. A flip rate of 0.30 says "30% of predictions
> would change if we toggled the protected attribute, holding other
> features as recorded." Whether that is acceptable, lawful, or causal
> is for the human reviewer (and possibly a domain causal graph) to
> decide.

The CLI does not ship this in 1.1; the surface is library-only, with a
proposed flag for the next CLI synthesis pass (see "Proposed CLI
surface" below).

## Proposed CLI surface (deferred to synthesis pass)

```bash
equiml audit data.csv \
    --target income --sensitive gender race \
    --counterfactual \
    --counterfactual-top-k 5 \
    --output audit.json
```

### Flags

- `--counterfactual` (bool flag) — when set, run the naive counterfactual
  audit against each declared sensitive feature and the proxy ranker
  against every non-protected, non-target column.
- `--counterfactual-top-k <int>` (default `10`) — number of proxies to
  surface per protected attribute.

### What it adds to the metrics dict

A new top-level block keyed `counterfactual` is appended to both
`baseline` and `fair` in the audit JSON:

```json
"counterfactual": {
  "<sensitive_feature>": {
    "flip_rate": 0.18,
    "mean_prediction_shift": 0.092,
    "n_samples": 600,
    "notes": [],
    "proxy_features": [
      {
        "feature_name": "salary_history",
        "flip_rate_with": 0.18,
        "flip_rate_without": 0.66,
        "proxy_strength": 0.48
      }
    ]
  }
}
```

The block lives at the same level as `per_sensitive` and follows the
same per-feature shape, so `equiml verify` and `equiml card` can render
it without further plumbing. The model card gets a "Counterfactual
sensitivity" subsection under Metrics.

### Exit codes

No new exit codes. The counterfactual block is reported, not gated, in
v1. Adding a `counterfactual_flip_rate` gate to `fairness.yaml` is
future work (see below).

## Alternatives considered

### Full Kusner et al. causal counterfactual

Pro: the *correct* answer to the counterfactual fairness question. Pro:
the academic baseline.

Con: requires a structural causal model the user must hand-specify and
keep in sync with the data schema. The dependency surface (DoWhy,
EconML, or a hand-rolled SCM solver) is heavy. Bad SCM input gives a
causally labelled wrong answer, which is worse than no answer. The
audit tool cannot validate the SCM the user supplies.

**Verdict:** out of scope for v1. Future work behind an opt-in dep
(`equiml[causal]`).

### DoWhy / EconML for the proxy step

Pro: well-tested causal effect estimators. Pro: established in the
field.

Con: both pull in `pgmpy`, `networkx`, `causalml`, `pymc`, and a
patchwork of solver backends. EquiML's whole pitch is a small, audited
core; pulling in an SCM stack for one feature breaks that. The proxy
detection use case (rank features by how much each carries the
protected signal) does not need full causal effect estimation — it
needs a sensitivity score.

**Verdict:** rejected for the proxy step. The neutralise-and-re-flip
approach lands the same diagnostic without the dep.

### Permutation importance as a fairness signal

Pro: permutation importance is already in scikit-learn. Cheap.

Con: permutation importance answers "how much does the model's
*accuracy* depend on this feature?" — not "how much does this feature
carry the *protected* signal?" A feature can be highly predictive
without correlating with the protected attribute (and vice versa). The
two questions are different, and folding them together hides the part
the audit cares about.

**Verdict:** rejected. The neutralise-and-re-flip variant answers the
specifically-fairness question.

### Naive counterfactual as a single number on the audit dashboard

Pro: maximum simplicity.

Con: a single flip-rate number, with no breakdown by sensitive feature
or proxy attribution, encourages the bad reading "this number is the
counterfactual fairness score." We deliberately ship a structured
result that makes the proxy ranking the load-bearing artefact, not the
flip rate alone.

**Verdict:** structured result, not a scalar.

## Consequences

### What gets easier

- An engineer reviewing a single denial can run the audit, look at the
  flip rate, and get a defensible answer to "would this have decided
  differently for the same person of a different gender?"
- Proxy detection ranks the features the model is laundering the
  protected signal through. That ranking is actionable: the team can
  remove or constrain the top proxies in the next training cycle and
  re-audit.
- The output is library-first, so a notebook user gets the API in v1
  and the CLI flag arrives once the synthesis pass lands.

### What gets harder

- Flip rate is a number, not a verdict. It will get misread as "the
  fairness score." The RFC and the docstring are explicit; the model
  card section will lead with the limitation.
- Proxy strength reads inverted to first instinct: a high strength
  means *neutralising* the candidate raised the flip rate (because the
  model lost a channel for hiding behind the proxy). The ProxyFeature
  docstring is explicit about the direction.
- Multi-class protected attributes average across cycles. That hides
  per-level differences (`gender ∈ {M, F, NB}` could be highly flippy
  on M↔NB but stable on F↔NB). v2 will expose per-level matrices.

### Backward compatibility

Net new module, no existing API changes. The CLI is unchanged in 1.1.

## Tested with

- `tests/test_counterfactual.py` (12 tests):
  - Flip rate ≈ 0 when the model never saw the protected attribute.
  - Flip rate ≈ 1 on a synthetic dataset where the protected attribute
    is the only useful signal.
  - `mean_prediction_shift` reported when `predict_proba` is available;
    `None` (with a note) when it is not.
  - Multi-class protected attribute: cycles through levels and emits
    the multi-class note.
  - Proxy detection: a perfect-copy `proxy` column ranks first; an
    unrelated noise column ranks last.
  - `proxy_strength` has the correct sign (positive for proxies).
  - Candidate-feature filter excludes the protected column even when
    passed explicitly.
  - Unknown candidate features raise `KeyError`.
  - `asdict(result)` round-trips, including nested `proxy_features`.
  - Single-level protected attribute raises `ValueError`.

- `examples/counterfactual_demo.py` — synthetic hiring dataset where
  `salary_history` is engineered as a near-copy of `gender`. The script
  asserts that `salary_history` is the top proxy.

## Future work (not in this RFC)

- **Full causal CF behind an opt-in dep**: `pip install equiml[causal]`
  pulls in DoWhy and exposes
  `equiml.counterfactual.compute_causal_counterfactual(model, X, scm)`
  for teams that have a structural causal model and want the real
  thing.
- **Intersectional counterfactual**: flip multiple protected attributes
  jointly (e.g. gender × race) and report the joint flip rate. Requires
  a 2-D output that the result dataclass does not carry today.
- **Per-level flip matrices for multi-class attributes**: report
  `{from: {to: flip_rate}}` instead of a single averaged scalar.
- **`counterfactual_flip_rate` gate in `fairness.yaml`**: e.g.
  `flip_rate: { max: 0.05, severity: error }`. Requires deciding what
  threshold means in a sensitivity-analysis context; deliberately
  punted to the next policy RFC.
- **HTML report integration**: a "Counterfactual sensitivity" section
  in the report template with the proxy table rendered.
- **CLI flag landing**: the `--counterfactual` and
  `--counterfactual-top-k` flags spec'd above ship in the next CLI
  synthesis pass.
