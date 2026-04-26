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

1. **Bias Detection** - Measures demographic parity, equalized odds, and disparate impact across sensitive groups
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

## Scope and limitations

- Tabular classification only. No text, image, or time-series models.
- Sensitive features must be declared explicitly — no automatic detection.
- Fairness metrics assume train and test distributions are i.i.d.
- The fair model always reduces accuracy. EquiML makes the tradeoff visible; the decision is yours.

## License

MIT
