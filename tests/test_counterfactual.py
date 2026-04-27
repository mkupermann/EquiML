"""Tests for naive counterfactual fairness + proxy detection (RFC 0004)."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from equiml.counterfactual import (
    CounterfactualResult,
    ProxyFeature,
    compute_counterfactual_audit,
    compute_proxy_features,
)


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _fit_logreg(X: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X, y)
    return model


# --- Sensitivity behaviour -------------------------------------------------


def test_flip_rate_zero_when_protected_attribute_unused(rng):
    """If the model never sees the protected attribute, flipping it must
    not change predictions. This is the trivial-sanity baseline."""
    n = 400
    age = rng.normal(40, 10, n)
    hours = rng.normal(40, 8, n)
    gender = rng.choice([0, 1], n)
    # y depends on age + hours only.
    y = ((age + hours) > 80).astype(int)

    X_with_gender = pd.DataFrame({"age": age, "hours": hours, "gender": gender})
    # Train WITHOUT gender — the model literally cannot use it.
    model = _fit_logreg(X_with_gender[["age", "hours"]], y)

    # But run the audit on a frame that contains gender, while the model
    # only consumes age+hours. We achieve this by predicting through a
    # column subset.
    class _Wrapped:
        def __init__(self, inner):
            self.inner = inner

        def predict(self, X):
            return self.inner.predict(X[["age", "hours"]])

        def predict_proba(self, X):
            return self.inner.predict_proba(X[["age", "hours"]])

    result = compute_counterfactual_audit(
        _Wrapped(model), X_with_gender, "gender",
    )
    assert result.flip_rate == pytest.approx(0.0, abs=1e-9)
    assert result.n_samples == n


def test_flip_rate_high_when_protected_attribute_is_only_predictor(rng):
    """When the model's only useful signal is the protected attribute,
    flipping it should change every (or near every) prediction."""
    n = 400
    gender = rng.choice([0, 1], n)
    noise = rng.normal(0, 0.01, n)  # dominated by gender
    # y = gender (with tiny jitter)
    y = gender.copy()

    X = pd.DataFrame({"gender": gender, "noise": noise})
    model = _fit_logreg(X, y)

    result = compute_counterfactual_audit(model, X, "gender")
    # Should be ~1.0; tolerate a few percent drift from the LR boundary.
    assert result.flip_rate > 0.95
    assert result.mean_prediction_shift is not None
    assert result.mean_prediction_shift > 0.5


def test_predict_proba_shift_reported_when_available(rng):
    n = 200
    a = rng.normal(0, 1, n)
    s = rng.choice([0, 1], n)
    y = ((a + s) > 0).astype(int)
    X = pd.DataFrame({"a": a, "s": s})
    model = _fit_logreg(X, y)

    result = compute_counterfactual_audit(model, X, "s")
    assert result.mean_prediction_shift is not None
    assert result.mean_prediction_shift > 0.0


def test_mean_shift_none_when_predict_proba_unsupported(rng):
    """Models that raise on predict_proba (e.g. ExponentiatedGradient)
    should produce a None shift, not a crash."""
    class _NoProbaModel:
        def predict(self, X):
            return (X["s"].to_numpy() > 0).astype(int)

        # Intentionally no predict_proba.

    n = 50
    X = pd.DataFrame({"s": np.r_[np.zeros(n // 2), np.ones(n // 2)]})
    result = compute_counterfactual_audit(_NoProbaModel(), X, "s")
    assert result.mean_prediction_shift is None
    assert any("predict_proba" in n for n in result.notes)


# --- Multi-class protected attribute ---------------------------------------


def test_multi_class_protected_attribute_cycles_levels(rng):
    """A 3-level protected attribute must cycle and average flip rates,
    not silently treat it as binary."""
    n = 600
    race = rng.choice([0, 1, 2], n)
    other = rng.normal(0, 1, n)
    # y depends mostly on race
    y = (race == 0).astype(int)

    X = pd.DataFrame({"race": race, "other": other})
    model = _fit_logreg(X, y)

    result = compute_counterfactual_audit(model, X, "race")
    assert 0.0 < result.flip_rate <= 1.0
    assert any("Multi-class" in n for n in result.notes)


# --- Proxy detection -------------------------------------------------------


def test_proxy_feature_is_ranked_first_when_perfect_copy(rng):
    """A non-protected feature that is a perfect copy of the protected
    attribute should rank first; an unrelated noise column should rank
    near the bottom."""
    n = 600
    gender = rng.choice([0, 1], n)
    proxy = gender.copy()  # perfect copy
    noise = rng.normal(0, 1, n)
    y = gender.copy()

    X = pd.DataFrame({"gender": gender, "proxy": proxy, "noise": noise})
    model = _fit_logreg(X, y)

    proxies = compute_proxy_features(model, X, "gender", top_k=10)
    names = [p.feature_name for p in proxies]
    # `proxy` should be the strongest proxy
    assert names[0] == "proxy"
    # `noise` should be the weakest (or absent if top_k clipped, but here
    # top_k=10 > 2 candidates)
    proxy_strength = {p.feature_name: p.proxy_strength for p in proxies}
    assert proxy_strength["proxy"] > proxy_strength["noise"]


def test_proxy_feature_strengths_have_expected_sign(rng):
    n = 400
    gender = rng.choice([0, 1], n)
    proxy = gender.copy()
    y = gender.copy()
    X = pd.DataFrame({"gender": gender, "proxy": proxy})
    model = _fit_logreg(X, y)

    proxies = compute_proxy_features(model, X, "gender")
    found = {p.feature_name: p for p in proxies}
    # proxy_strength is flip_rate_with - flip_rate_without; perfect proxy
    # should yield a large positive strength.
    assert found["proxy"].proxy_strength > 0.4


def test_candidate_features_filter_excludes_protected_column(rng):
    n = 100
    gender = rng.choice([0, 1], n)
    other = rng.normal(0, 1, n)
    y = gender.copy()
    X = pd.DataFrame({"gender": gender, "other": other})
    model = _fit_logreg(X, y)

    # Even if the caller passes the protected column as a candidate, it
    # must be filtered out of the result — proxy detection compares
    # *non-protected* features.
    proxies = compute_proxy_features(
        model, X, "gender",
        candidate_features=["gender", "other"],
    )
    assert all(p.feature_name != "gender" for p in proxies)


def test_unknown_candidate_features_raise_keyerror(rng):
    X = pd.DataFrame({"gender": [0, 1, 0, 1], "x": [0.0, 1.0, 0.0, 1.0]})
    y = np.array([0, 1, 0, 1])
    model = _fit_logreg(X, y)
    with pytest.raises(KeyError):
        compute_proxy_features(
            model, X, "gender", candidate_features=["does_not_exist"],
        )


# --- Result-shape contracts -----------------------------------------------


def test_result_dataclass_round_trips_via_asdict(rng):
    n = 100
    gender = rng.choice([0, 1], n)
    other = rng.normal(0, 1, n)
    y = (gender + (other > 0).astype(int)).clip(0, 1)
    X = pd.DataFrame({"gender": gender, "other": other})
    model = _fit_logreg(X, y)

    result = compute_counterfactual_audit(model, X, "gender")
    result.proxy_features = compute_proxy_features(model, X, "gender")
    payload = asdict(result)
    assert isinstance(payload, dict)
    assert "flip_rate" in payload
    assert "proxy_features" in payload
    # proxy_features serialises to a list of dicts
    assert all(isinstance(p, dict) for p in payload["proxy_features"])
    assert all("proxy_strength" in p for p in payload["proxy_features"])


def test_invalid_sensitive_feature_raises(rng):
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    class _M:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    with pytest.raises(KeyError):
        compute_counterfactual_audit(_M(), X, "not_a_column")


def test_single_level_protected_attribute_raises(rng):
    X = pd.DataFrame({"s": [1, 1, 1, 1], "x": [0.1, 0.2, 0.3, 0.4]})
    class _M:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    with pytest.raises(ValueError):
        compute_counterfactual_audit(_M(), X, "s")
