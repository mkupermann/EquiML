<!-- TODO(maintainer): update the GitHub repo "About" description (set in
     the GitHub UI, not here) to: "A focused CLI for fairness audits on
     tabular ML — wraps fairlearn, scikit-learn, and SHAP." -->

# EquiML

[![CI](https://github.com/mkupermann/EquiML/actions/workflows/ci.yml/badge.svg)](https://github.com/mkupermann/EquiML/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

Quick fairness audits for ML datasets from the command line.

EquiML wraps [fairlearn](https://fairlearn.org/), [SHAP](https://shap.readthedocs.io/), and [scikit-learn](https://scikit-learn.org/) into a single pipeline: load data, detect bias, train a fair model, compare results.

<!-- TODO(maintainer): two sentences in your own voice. Why fairness, why now,
     what client pattern made you tired of rebuilding this glue. Don't
     ghostwrite this paragraph — it either lands as conviction or it
     weakens the rest of the README. -->

## Why EquiML

Most fairness tooling is scattered — one library for metrics, another for constraints, a third for explainability. EquiML is the three-command pipeline I kept rebuilding on every audit. It is opinionated by design: one dataset in, a report out, honest comparison between baseline and fairness-constrained model.

It is not a new algorithm. It is a way to stop copying the same glue code.

## Install

```bash
pip install -e .
```

Tested on Python 3.9–3.12 (macOS / Linux). On Apple Silicon or Windows, SHAP wheels can need build tools; if `pip install` errors, run `pip install --upgrade pip setuptools wheel` first and retry.

Project files: [`SECURITY.md`](SECURITY.md) · [`CHANGELOG.md`](CHANGELOG.md) · [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quickstart

Run a fairness audit on the UCI Adult dataset in under a minute:

```bash
python examples/adult_census_audit.py
```

The script downloads ~5,000 rows of the [Adult census](https://archive.ics.uci.edu/dataset/2/adult), trains a baseline logistic-regression model plus a fairness-constrained variant, and writes:

- `examples/adult_audit.json` — metrics plus a `_meta` block (library versions, random seed, command args) suitable for a CI artefact or compliance evidence pack.
- `examples/adult_audit.html` — side-by-side baseline-vs-fair report.

Expected console output:

```text
PERFORMANCE (Baseline / Fair Model)
  Accuracy:  81.7%  /  81.6%
  F1-Score:  80.7%  /  80.3%

FAIRNESS (lower = fairer)
  Demographic Parity:  0.205  /  0.041
  Equalized Odds:      0.338  /  0.285
```

### How to read this

Demographic parity is the gap in selection rates between sensitive groups, scored 0–1. `0.205` means the favoured group is selected 20 percentage points more often; `0.041` is small but not zero. The fair model trains under a `DemographicParity` constraint via fairlearn's `ExponentiatedGradient`, so it pulls hard on that gap. Equalized Odds (the gap in true and false positive rates) only moves from `0.338` to `0.285` — the constraint optimised demographic parity, not equalized odds, and a different constraint would trade differently. Adult itself has [documented label and selection bias](https://arxiv.org/abs/2108.04884), so treat this run as a tutorial, not a model for new work.

### Run it on your CSV

```bash
equiml audit your_data.csv --target <column you predict> --sensitive <protected column>
```

The target must be binary. The sensitive feature can be a single column or several (`--sensitive gender race`); each is audited individually and per-feature metrics land under `metrics["per_sensitive"][<name>]` in the JSON.

### What to do with the output

Open `adult_audit.html` and read the side-by-side panels. The JSON's `_meta` block is your audit-trail artefact: commit it next to your model card, attach it to a CI run, or hand it to a Risk colleague as evidence under EU AI Act Art. 15. EquiML reports numbers; whether to retrain, monitor via `BiasMonitor`, or escalate is a domain decision.

### Troubleshooting

- **Audit can't reach OpenML or UCI on first run.** The example needs network access to one of the two; behind a corporate proxy, set `HTTPS_PROXY`. The CSV caches at `examples/adult.csv` after a successful run, so subsequent runs are offline.
- **Audit is slow on a large dataset.** SHAP's `PermutationExplainer` scales poorly on tree models. Subsample to ~5k rows for a fast demo, full data for a real audit.
- **`predict_proba` fails on the fair model.** That's by design: `fairlearn.reductions.ExponentiatedGradient` is a randomised classifier and probability averaging across its predictors is mathematically incorrect. Use `predict()` and report fairness from hard predictions.

## Usage

### CLI

```bash
# Run a fairness audit
equiml audit data.csv --target income --sensitive gender race

# Generate an HTML report
equiml audit data.csv --target income --sensitive gender --report audit.html

# Save detailed metrics as JSON
equiml audit data.csv --target income --sensitive gender --output metrics.json

# Use a different algorithm
equiml audit data.csv --target income --sensitive gender --algorithm random_forest
```

### Python API

```python
from equiml import Data, Model, EquiMLEvaluation

# Load and preprocess
data = Data(dataset_path="data.csv", sensitive_features=["gender"])
data.load_data()
data.preprocess(target_column="income",
                numerical_features=["age", "hours_per_week"],
                categorical_features=["gender"])
data.split_data()

# Train with fairness constraint
model = Model(algorithm="logistic_regression", fairness_constraint="demographic_parity")
model.train(data.X_train, data.y_train, sensitive_features=data.X_train["gender"])

# Evaluate
evaluator = EquiMLEvaluation()
metrics = evaluator.evaluate(model, data.X_test, data.y_test,
                            sensitive_features=data.X_test["gender"])
print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Demographic Parity: {metrics['demographic_parity_difference']:.3f}")
```

### Monitoring

```python
from equiml import BiasMonitor, DriftDetector

# Monitor predictions for bias drift
monitor = BiasMonitor(sensitive_features=["gender"])
result = monitor.monitor_predictions(predictions, sensitive_df)

# Detect data drift
detector = DriftDetector(reference_data=training_data)
drift = detector.detect_drift(new_data)
```

## What it does

1. **Bias Detection** - Surfaces fairness metrics (demographic parity, equalized odds, disparate impact) across sensitive groups. You decide whether the disparity is acceptable for your domain — the tool reports numbers, not verdicts.
2. **Fair Training** - Trains models with fairness constraints using fairlearn's ExponentiatedGradient
3. **Comparison** - Shows baseline vs. fair model side-by-side so you see the accuracy/fairness tradeoff
4. **Explainability** - SHAP and LIME explanations for model decisions
5. **Monitoring** - Track bias metrics and data drift over time
6. **Reporting** - HTML reports with metrics and recommendations

## What it wraps

This is an opinionated pipeline, not a new algorithm. The actual ML work is done by:

| Capability | Library |
|-----------|---------|
| Fairness metrics and constraints | [fairlearn](https://fairlearn.org/) |
| Model explanations | [SHAP](https://shap.readthedocs.io/), [LIME](https://github.com/marcotcr/lime) |
| ML models | [scikit-learn](https://scikit-learn.org/) |
| Statistical tests | [scipy](https://scipy.org/), [statsmodels](https://www.statsmodels.org/) |

## How EquiML compares to fairlearn / aif360 / Aequitas

- **[fairlearn](https://fairlearn.org/)** is the underlying library. EquiML's job is to be a CLI you can pipe into CI; fairlearn is the toolbox you build with.
- **[aif360](https://aif360.res.ibm.com/)** (IBM) has a broader algorithm catalogue, a heavier install, and a more academic shape. Choose aif360 if you need pre-, in-, and post-processing variety beyond `ExponentiatedGradient` and reweighing.
- **[Aequitas](http://www.datasciencepublicpolicy.org/our-work/tools-guides/aequitas/)** (CMU/DSSG) is policy-audit-shaped, with group-disparity dashboards. It is often the right pick for public-sector audits.
- **EquiML** is an opinionated three-command pipeline, single-author, built for a specific audit cadence — not a research toolkit.

If you don't already know which of these you need, you probably need fairlearn or Aequitas, not EquiML.

## Where this fits in your governance framework

EquiML produces evidence; the assessment is a human and legal exercise. It is **not a Conformity Assessment**. The audit fits the **Measure** function of the **NIST AI Risk Management Framework** (Govern / Map / Measure / Manage) and does not cover Govern or Manage — those are policy and process work that sits outside the tool.

| EquiML artefact | Maps to | What it evidences |
|---|---|---|
| `metrics.demographic_parity_difference` etc. (JSON) | EU AI Act Art. 15 §3 | Group-disparity testing was performed on this model. Not that the result is acceptable — that is a domain decision. |
| `_meta` block (JSON) | EU AI Act Art. 12 (record keeping); ISO/IEC 42001 §8.3 | Reproducibility metadata for a single audit run: versions, seed, dataset path, args. |
| HTML report | ISO/IEC 42001 §9.1 (monitoring) | Human-reviewable artefact for an AI-management-system review cycle. |
| Audit run itself | NIST AI RMF Measure | One Measure-function activity; does not cover Govern or Manage. |

The crosswalk is a starting point for your evidence pack, not a substitute for legal review of which clauses apply to your model and jurisdiction.

## Supported algorithms

- `logistic_regression` (default)
- `random_forest`
- `ensemble` (voting classifier)

## Project structure

```
equiml/
  __init__.py      # Package exports
  cli.py           # CLI entry point (equiml audit ...)
  data.py          # Data loading, preprocessing, bias mitigation
  model.py         # Model training with fairness constraints
  evaluation.py    # Metrics: performance, fairness, robustness, explainability
  monitoring.py    # BiasMonitor, DriftDetector
  reporting.py     # HTML report generation
  report_template.html
tests/
  test_audit_pipeline.py
```

## What this tool does not do

- Not a legal opinion or regulatory verdict.
- Not jurisdiction-specific. Fairness definitions vary by domain and by law.
- No intersectional fairness analysis (yet). Bias at the cross-product of two protected attributes is not assessed.
- Assumes train and test are i.i.d. Not appropriate when distributions shift.
- Tabular classification only. No text, image, time-series, or LLM evaluation.

## About the author

Built by Michael Kupermann at [Kupermann Consulting](https://kupermann.com).
Companion writing on Medium: [Cortex HDC](URL_HERE), [FIRE Score](URL_HERE).
Issues, ideas, and collaboration enquiries welcome.

## License

MIT
