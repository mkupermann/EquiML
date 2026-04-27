# RFC 0006 — Group-Uncertainty Aware Fairness Intervals

| Field | Value |
|---|---|
| Status | Implemented (1.2.0-dev) |
| Author | Michael Kupermann |
| Date | 2026-04-26 |
| Reviewers | (open for comment) |

## Context

`equiml audit` today reports fairness as point estimates: "demographic parity difference is 0.04." That number is treated as load-bearing — it is what compliance pastes into a memo, what a CI gate fails on, and what a regulator will read in the audit pack. It is also wrong, in a specific and predictable way: it ignores three sources of uncertainty in the protected attribute itself.

1. **Sample uncertainty.** The audit runs on one finite test split. Re-run it on a different random split of the same data and the metric moves. With 500 test rows, a DP of 0.04 is not statistically distinguishable from a DP of 0.10.

2. **Missingness.** Self-reported protected attributes (race, ethnicity, gender) are missing for some fraction of rows in any real dataset. The audit either drops those rows (silently biasing the result toward respondents) or ignores the missingness altogether.

3. **Noisy labels.** Even when supplied, protected attributes are noisy: the observed group label may differ from the "true" label that should govern the fairness analysis. Surveys collapse multi-racial individuals into a single category; gender is binarised; self-ID flips between application forms. Chen, Johansson & Sontag (2018, "Why is my classifier discriminatory?") and Kallus & Zhou (2018, "Residual unfairness in fair machine learning from prejudiced data") both show that ignoring these sources can flip the sign of the disparity estimate, not just shrink its CI.

Today, `equiml audit` treats all three as zero. The load-bearing claim "DP = 0.04" hides "actually somewhere between 0.01 and 0.09 once you account for missingness." That is the gap this RFC closes.

## Decision

EquiML 1.2 ships:

1. A new module `equiml/uncertainty.py` with three orthogonal simulators that compose by stacking:
   - `bootstrap_fairness_intervals` — non-parametric bootstrap for sample uncertainty.
   - `impute_and_bootstrap` — multiple imputation + bootstrap for partly-missing protected attributes.
   - `noise_model_intervals` — Monte Carlo over a user-supplied noise matrix for label-noise uncertainty.

2. Two dataclasses:
   - `MetricInterval(point, lower, upper, confidence)` — the unit of reporting. `__str__` renders `"0.04 [0.01, 0.09] (95%)"`.
   - `UncertaintyResult` — a bundle of intervals across DP / EO / EOpp, plus simulator metadata (`n_bootstrap`, `n_imputations`, `missing_fraction`, `notes`).

3. A demo script (`examples/uncertainty_demo.py`) showing the three simulators on a synthetic dataset where 25% of `gender` is missing.

The CLI surface is **proposed but not implemented in this RFC** — see [Future work](#future-work).

## Three sources, three simulators, one dataclass

The simulators are independent. A single audit can report any subset.

### Bootstrap

Standard non-parametric bootstrap. For each of `n_bootstrap` resamples:

1. Draw `n` row indices with replacement.
2. Compute DP / EO / EOpp on the resample.
3. Skip the resample if it collapses to a single sensitive group.

The percentile band on the resulting population is the interval.

```python
res = bootstrap_fairness_intervals(
    y_true, y_pred, sensitive_features,
    n_bootstrap=1000, confidence=0.95, random_state=42,
)
print(res["demographic_parity_difference"])   # "0.04 [0.01, 0.09] (95%)"
```

### Multiple imputation

For partly-missing sensitive attributes. For each of `n_imputations`:

1. Fill NaN sensitive values by sampling from the empirical distribution — marginal if no covariates are passed; conditional on a coarse stratification of `features` if they are.
2. Bootstrap fairness metrics on the imputed dataset.

The bootstrap draws are pooled across imputations and the percentile band is taken on the pool. This is the simpler-than-Rubin pooling we explain in [Alternatives considered](#alternatives-considered).

```python
res = impute_and_bootstrap(
    y_true, y_pred, sensitive_features_with_nan,
    features=covariates_dataframe,   # optional; enables conditional imputation
    n_imputations=20, n_bootstrap_per_impute=200,
)
```

### Noise model

For known label noise. The user supplies a `P(observed | true)` matrix as a DataFrame, dict-of-dicts, or 2-D ndarray. The simulator inverts via Bayes' rule with the observed marginal as the prior, then for each simulation draws `true` labels per row and recomputes fairness against them.

```python
matrix = pd.DataFrame(
    [[0.9, 0.1], [0.1, 0.9]],
    index=["A", "B"], columns=["A", "B"],
)
res = noise_model_intervals(
    y_true, y_pred, sensitive_observed,
    error_rate_matrix=matrix, n_simulations=1000,
)
```

### Composition

The three are orthogonal. A future `compose_intervals(...)` API can stack them — e.g. impute + simulate noise + bootstrap — by feeding each simulator's draws into the next. We do not ship the composer in 1.2 because we want to see how teams use the individual simulators first; over-eager composition tends to produce intervals so wide they are unactionable.

## Honest naming of limitations

Each simulator's docstring names what it does NOT cover.

- **Bootstrap** assumes the test sample is representative of the deployment distribution. If the test set was sampled with a different selection bias than deployment, the interval is centred on the wrong number.

- **Multiple imputation** assumes the protected attribute is **missing at random (MAR)** conditional on the supplied features (or **MCAR** if no features are supplied). If missingness correlates with the outcome (NMAR — e.g. people who would face discrimination are more likely to skip the question), this interval is optimistic. The audit pack should disclose what was assumed.

- **Noise model** trusts the user's `error_rate_matrix`. We do not estimate the matrix from data — that requires auxiliary ground truth which by definition the audit does not have. A wrong matrix produces a confidently wrong interval.

These are not footnotes; they appear in the module docstring and in `notes` on every `UncertaintyResult`.

## Proposed CLI surface (not implemented in this RFC)

The synthesis-pass change to `equiml/cli.py` will add:

```bash
equiml audit data.csv \
    --target income --sensitive gender race \
    --uncertainty \
    --n-bootstrap 1000 \
    --confidence 0.95 \
    --output audit.json
```

Behaviour:

- `--uncertainty` adds an `uncertainty` block to the audit JSON's `metrics` dict, one `MetricInterval` per fairness metric.
- The JSON shape becomes `{"baseline": {..., "uncertainty": {"demographic_parity_difference": {"point": 0.04, "lower": 0.01, "upper": 0.09, "confidence": 0.95}, ...}, ...}, ...}`.
- The CLI **auto-detects** missing values in any column listed in `--sensitive`. If `>0` rows have `NaN` in a sensitive column, it switches from `bootstrap_fairness_intervals` to `impute_and_bootstrap` and adds a note to the JSON saying so.
- `--n-bootstrap` defaults to `1000`; `--confidence` defaults to `0.95`.
- The console output appends a single `UNCERTAINTY` block under the existing `FAIRNESS` block, rendering each interval via `MetricInterval.__str__`.
- The HTML report renders intervals as a third column next to the baseline / fair point estimates.
- `equiml verify` and the model-card generator (`equiml card`) read `metrics.uncertainty` if present and surface intervals alongside point estimates.

The noise-model simulator does **not** get a CLI flag in 1.2 — the noise matrix is too domain-specific for a flag. Library users construct it explicitly. A future `--noise-matrix path/to/matrix.yaml` is an option once we see what teams put there.

Exit codes are unchanged. Policy gates continue to evaluate against the point estimate; an `--uncertainty-policy` mode that gates on the interval bound (instead of the point) is [Future work](#future-work).

## Alternatives considered

### Bayesian credible intervals via PyMC

Pro: principled posterior over the metric; pools missingness, noise, and sample variance into one model.
Con: heavy dependency for one feature. PyMC drags in PyTensor, which alone doubles the wheel size. Sampler tuning is a per-dataset chore; teams that wanted that would already be using PyMC. The audience for `equiml audit` is engineers who want a number for a CI gate, not a posterior plot.

**Verdict:** rejected. Frequentist bootstrap + percentile bands give the same interval shape with no extra dependency.

### Parametric CI via the delta method

Pro: closed-form, no resampling.
Con: the delta method works for a single metric expressed as a smooth function of summary statistics. Fairness gaps (DP, EO, EOpp) are **non-smooth** at the boundary where group membership flips — they are differences of conditional rates, not means. The delta-method standard error misses the resampling variance from group membership and produces intervals that are too tight.

**Verdict:** rejected. The bootstrap captures the joint variation that closed-form does not.

### Rubin's rules for pooling across imputations

Pro: textbook standard for multiple imputation. Decomposes total variance into within- and between-imputation components.
Con: Rubin's rules assume the within-imputation estimator is approximately normal. Fairness gaps are bounded in `[-1, 1]` and skewed near the boundaries; the normal approximation is poor exactly when the gap matters most. Pooling the bootstrap draws across imputations (what we do) gives the correct distribution shape without the normality assumption.

**Verdict:** rejected. Pool the percentiles, not the moments.

### Conformal prediction intervals

Pro: distribution-free, finite-sample valid.
Con: conformal intervals are constructed for **predictions**, not for **summary statistics of predictions**. Wrapping a fairness metric in a conformal interval is a research project, not an audit feature.

**Verdict:** rejected. Possibly revisited if the conformal-fairness literature settles.

### Skip the whole thing and add a confidence-interval column to the existing fairlearn output

Pro: minimal code.
Con: fairlearn does not ship intervals. We would still be writing the bootstrap loop. The dataclass + module is the same work either way.

**Verdict:** the dataclass/module is the right home. Fairlearn might adopt this upstream; we are not blocking on that.

## Consequences

### What gets easier

- Regulator-defensible reporting. "DP = 0.04" becomes "DP = 0.04 with 95% interval [0.01, 0.09]" — the interval makes the claim auditable.
- Honest "we do not know" instead of false precision. A DP of 0.04 with interval `[-0.05, 0.13]` says the test set is too small to make any claim; a DP of 0.04 with interval `[0.03, 0.05]` is the load-bearing one.
- Procurement reviewers stop asking "is this number stable?" because the interval answers it.

### What gets harder

- Audit JSONs grow. An `uncertainty` block adds three intervals × four numbers per fairness metric. The compute cost is `n_bootstrap * O(metric_compute)` — for `n=1000` bootstrap and `n_test=10k` rows, each fairness metric runs in seconds, so the audit goes from `~5s` to `~30s`. A future flag could lower `n_bootstrap` for fast feedback loops.
- Teams have to choose `n_bootstrap` and `confidence`. We supply defaults (`1000`, `0.95`); the defaults are documented in the RFC and surface as CLI flags.
- The noise-model simulator requires the user to supply a noise matrix. We do not invent a default — there is no domain-neutral one. Teams that do not have a matrix can run only the bootstrap and imputation simulators.

### Backward compatibility

Net new feature. The dataclasses are new; no existing CLI surface or JSON shape changes until the synthesis-pass `--uncertainty` flag lands. Without the flag, every 1.1.x audit invocation produces identical results.

## Tested with

- Bootstrap interval contains the point estimate on a fully-observed dataset.
- Interval widens as test set shrinks (sample-size effect).
- Reproducibility: same seed → identical intervals.
- Empirical coverage: a 95% interval contains 95% of bootstrap draws to within 2 percentage points.
- 30%-missing protected attribute: imputation interval is wider than the oracle bootstrap that knew the truth (isolating the missingness effect from the sample-size effect).
- Edge cases: empty input → `ValueError`; constant predictions → DP = 0 with zero-width interval; noise matrix with zero rows is handled.
- Pandas Series and numpy ndarray inputs both accepted.

## Future work (not in this RFC)

- **CLI integration.** The `--uncertainty` flag described above is a synthesis-pass change to `equiml/cli.py`.
- **Composition API.** A `compose_intervals(simulators=[...])` that stacks the three sources properly (impute → noise → bootstrap) and reports a single interval that integrates all three.
- **Per-feature missingness models.** Today, missingness is treated identically across features. In real data, race is missing for different reasons than gender; per-feature MAR models would tighten the imputation interval where appropriate.
- **Intersectional uncertainty.** The simulators today work on a single sensitive feature. Intersectional fairness (race × gender) needs intervals on the joint distribution; the bootstrap generalises but the imputation does not without additional structure.
- **Differential-privacy-aware sensitive attributes.** When the protected attribute is supplied through a DP mechanism (e.g. the US Census 2020 noise infusion), the noise matrix is known and known-large. A first-class adaptor would consume the DP mechanism's noise spec directly.
- **`--uncertainty-policy` mode.** Gate CI on the *upper bound* of the interval rather than the point estimate. Conservative; will fail more runs; that is the regulator-defensible choice when the test set is small.
- **NMAR diagnostics.** We assume MAR. A diagnostic that flags when the missingness pattern is suspiciously correlated with the outcome (e.g. via a Heckman-style two-stage regression) would warn the user before they trust an MI-based interval.

## References

- Chen, I., Johansson, F. D., & Sontag, D. (2018). *Why is my classifier discriminatory?* NeurIPS 2018.
- Kallus, N., & Zhou, A. (2018). *Residual unfairness in fair machine learning from prejudiced data.* ICML 2018.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys.* Wiley.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
