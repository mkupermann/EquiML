"""``Model`` API contract tests.

Locks Teammate 1, tasks 1 and 2:

* ``predict_proba`` must raise ``NotImplementedError`` on a fairness-mitigated
  model. ``ExponentiatedGradient`` is a randomised classifier; arithmetic-mean
  averaging of the predictors' probabilities (the original implementation)
  is mathematically wrong, so the chosen contract is to raise rather than
  return a misleading number.
* ``cross_validate`` must raise on a mitigated model. The original code
  silently fell back to CV-ing the unconstrained estimator, which made the
  user think they were CV-ing the fair model.

Hard predictions (``predict``) and CV on the unconstrained baseline must
still work — the raise is targeted, not blanket.
"""

import numpy as np
import pandas as pd
import pytest

from equiml.data import Data
from equiml.model import Model


def _prepared_data(adult_sample) -> Data:
    data = Data(sensitive_features=["gender"])
    data.df = adult_sample
    data.preprocess(
        target_column="income",
        numerical_features=["age", "hours_per_week"],
        categorical_features=["gender"],
    )
    data.split_data()
    return data


def _sensitive_train(data: Data) -> pd.Series:
    cols = [c for c in data.X_train.columns if c == "gender" or c.startswith("gender_")]
    assert cols, "preprocessing should expose at least one gender column"
    return data.X_train[cols[0]]


class TestPredictProbaContract:
    def test_predict_proba_raises_on_mitigated_model(self, adult_sample):
        data = _prepared_data(adult_sample)
        sensitive = _sensitive_train(data)

        model = Model(
            algorithm="logistic_regression",
            fairness_constraint="demographic_parity",
        )
        model.train(data.X_train, data.y_train, sensitive_features=sensitive)

        with pytest.raises(NotImplementedError):
            model.predict_proba(data.X_test)

    def test_predict_still_works_on_mitigated_model(self, adult_sample):
        """Hard predictions must remain available after the raise contract."""
        data = _prepared_data(adult_sample)
        sensitive = _sensitive_train(data)

        model = Model(
            algorithm="logistic_regression",
            fairness_constraint="demographic_parity",
        )
        model.train(data.X_train, data.y_train, sensitive_features=sensitive)

        preds = model.predict(data.X_test)
        assert len(preds) == len(data.X_test)
        # Predictions must be 0/1 for binary classification
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_still_works_on_unconstrained_model(self, adult_sample):
        """The raise must be targeted at mitigators, not blanket."""
        data = _prepared_data(adult_sample)

        model = Model(algorithm="logistic_regression")
        model.train(data.X_train, data.y_train)

        proba = model.predict_proba(data.X_test)
        assert proba.shape == (len(data.X_test), 2)
        # Probabilities must sum to 1 per row.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestCrossValidateContract:
    def test_cross_validate_raises_on_mitigated_model(self, adult_sample):
        data = _prepared_data(adult_sample)
        sensitive = _sensitive_train(data)

        model = Model(
            algorithm="logistic_regression",
            fairness_constraint="demographic_parity",
        )
        model.train(data.X_train, data.y_train, sensitive_features=sensitive)

        with pytest.raises(NotImplementedError):
            model.cross_validate(
                data.X_train, data.y_train,
                sensitive_features=sensitive, cv=3,
            )

    def test_cross_validate_returns_scores_on_unconstrained_model(self, adult_sample):
        """Plain CV on the baseline must still return a usable scoring dict."""
        data = _prepared_data(adult_sample)

        model = Model(algorithm="logistic_regression")
        model.train(data.X_train, data.y_train)

        scores = model.cross_validate(data.X_train, data.y_train, cv=3)

        assert isinstance(scores, dict)
        # sklearn's cross_validate returns 'test_<scorer>' keys; check the
        # accuracy key is present and has one entry per fold.
        assert "test_accuracy" in scores, (
            f"cross_validate output missing 'test_accuracy'; got {list(scores)}"
        )
        assert len(scores["test_accuracy"]) == 3
