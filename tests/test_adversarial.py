"""Adversarial test cases for the audit pipeline.

This file is the **start of an adversarial test directory** for EquiML.
The intent is to grow it case-by-case (one scenario per commit), keeping
each case readable and debuggable rather than batching abstract property
checks. New cases should be added one at a time as they are designed.

Case 1: ``proxy_perfectly_correlated_with_protected``
    A non-protected feature (``proxy``) is a literal copy of the protected
    attribute. The label is biased: positive-class rate differs sharply
    between protected groups. The audit must still flag bias on the
    protected attribute. The test guards against a regression where a
    model "launders" sensitive information through a proxy and the audit
    fails to surface the resulting disparity.
"""

import numpy as np
import pandas as pd
import pytest

from equiml.data import Data
from equiml.evaluation import EquiMLEvaluation
from equiml.model import Model


def _adversarial_dataset(n: int = 300) -> pd.DataFrame:
    """Build a dataset with a perfect proxy and a strongly biased target."""
    rng = np.random.default_rng(2024)
    protected = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    # Perfect proxy: literal copy under another name. A model trained on
    # all features will pick this up freely.
    proxy = protected.copy()
    feature1 = rng.normal(size=n)
    feature2 = rng.normal(size=n)
    # Heavy bias on the target: protected=1 is ~80% positive,
    # protected=0 is ~20% positive.
    base_p = np.where(protected == 1, 0.8, 0.2)
    target = (rng.random(n) < base_p).astype(int)
    return pd.DataFrame({
        "feature1": feature1,
        "feature2": feature2,
        "proxy": proxy.astype(float),
        "protected": protected.astype(float),
        "target": target,
    })


def test_audit_flags_bias_when_proxy_is_perfectly_correlated():
    df = _adversarial_dataset()

    data = Data(sensitive_features=["protected"])
    data.df = df
    data.preprocess(
        target_column="target",
        numerical_features=["feature1", "feature2", "proxy", "protected"],
    )
    data.split_data(test_size=0.3)

    sensitive_test = data.X_test["protected"]

    model = Model(algorithm="logistic_regression")
    model.train(data.X_train, data.y_train)
    preds = model.predict(data.X_test)

    evaluator = EquiMLEvaluation()
    metrics = evaluator.evaluate(
        model, data.X_test, data.y_test,
        y_pred=preds,
        sensitive_features=sensitive_test,
    )

    dpd = abs(metrics["demographic_parity_difference"])
    # The injected disparity is ~0.6 in the population. After fitting the
    # model still has obvious group disparity; assert the audit catches it
    # well above the 0.1 "moderate" threshold.
    assert dpd > 0.1, (
        f"Audit failed to surface disparity for the protected attribute "
        f"despite a perfect proxy and a heavily biased target. "
        f"demographic_parity_difference = {dpd:.4f}"
    )
