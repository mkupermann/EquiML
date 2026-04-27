"""Tests for the feedback-loop simulator (RFC 0005)."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from equiml.model import Model
from equiml.simulation import (
    DecisionRule,
    FeedbackLoopSimulator,
    FeedbackRule,
    RoundResult,
    apply_decision_rule,
    apply_feedback,
)


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


def _unbiased_dataset(n: int = 600, seed: int = 0):
    """Two groups, identical underlying outcome distributions."""
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], n)
    feat1 = rng.normal(0, 1, n)
    feat2 = rng.normal(0, 1, n)
    # Outcome depends on features only — group has zero effect.
    logits = 0.8 * feat1 + 0.4 * feat2
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "group_A": (group == "A").astype(int)})
    return X, pd.Series(y, name="y"), pd.Series(group, name="group")


def _biased_dataset(n: int = 600, seed: int = 0):
    """Group A is historically over-selected even after controlling for features.

    Bias is moderate, not maximal: at threshold 0.5 a baseline LR model
    captures most of it on round 0, so we use a higher threshold in the
    feedback test to leave room for compounding.
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], n, p=[0.5, 0.5])
    feat1 = rng.normal(0, 1, n)
    feat2 = rng.normal(0, 1, n)
    # Group A gets a moderate additive boost — biased but not saturated.
    boost = np.where(group == "A", 0.6, -0.2)
    logits = 0.6 * feat1 + 0.3 * feat2 + boost
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "group_A": (group == "A").astype(int)})
    return X, pd.Series(y, name="y"), pd.Series(group, name="group")


def _split(X, y, s, test_frac=0.4, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1 - test_frac))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return (
        X.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        s.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        s.iloc[test_idx].reset_index(drop=True),
    )


def _factory():
    """Fresh unconstrained logistic-regression Model for each round."""
    return Model(algorithm="logistic_regression")


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------


def test_threshold_and_top_k_produce_different_selections():
    scores = np.array([0.1, 0.4, 0.55, 0.6, 0.9, 0.95])
    thr_idx = apply_decision_rule(scores, DecisionRule.THRESHOLD, threshold=0.5)
    topk_idx = apply_decision_rule(scores, DecisionRule.TOP_K, k=2)
    # Threshold > 0.5 catches indices 2, 3, 4, 5 (4 items).
    assert len(thr_idx) == 4
    # Top-2 picks the two highest scores.
    assert len(topk_idx) == 2
    assert set(topk_idx.tolist()) == {4, 5}
    # And they disagree on count.
    assert len(thr_idx) != len(topk_idx)


def test_threshold_rejects_invalid_value():
    with pytest.raises(ValueError):
        apply_decision_rule(np.array([0.1, 0.9]), DecisionRule.THRESHOLD, threshold=1.5)


def test_top_k_clamps_to_array_length():
    scores = np.array([0.1, 0.9])
    idx = apply_decision_rule(scores, DecisionRule.TOP_K, k=10)
    assert len(idx) == 2


# ---------------------------------------------------------------------------
# Feedback rules
# ---------------------------------------------------------------------------


def test_perfect_info_keeps_all_rows():
    X = pd.DataFrame({"f": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    s = pd.Series(["A", "A", "B", "B"])
    selected = np.array([0, 2])
    Xn, yn, sn = apply_feedback(FeedbackRule.PERFECT_INFO, X, y, s, selected)
    assert len(Xn) == 4
    assert list(yn) == [0, 1, 0, 1]


def test_selection_bias_drops_unselected():
    X = pd.DataFrame({"f": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    s = pd.Series(["A", "A", "B", "B"])
    selected = np.array([1, 3])
    Xn, yn, sn = apply_feedback(FeedbackRule.SELECTION_BIAS, X, y, s, selected)
    assert len(Xn) == 2
    assert list(yn) == [1, 1]
    assert list(sn) == ["A", "B"]


def test_selection_bias_with_no_selection_returns_empty():
    X = pd.DataFrame({"f": [1, 2]})
    y = pd.Series([0, 1])
    s = pd.Series(["A", "B"])
    Xn, yn, sn = apply_feedback(
        FeedbackRule.SELECTION_BIAS, X, y, s, np.array([], dtype=int)
    )
    assert len(Xn) == 0


# ---------------------------------------------------------------------------
# RoundResult
# ---------------------------------------------------------------------------


def test_round_result_round_trips_via_asdict():
    r = RoundResult(
        round_idx=0,
        metrics={"demographic_parity_difference": 0.04, "accuracy": 0.81},
        per_sensitive={"group": {"demographic_parity_difference": 0.04}},
        n_train=200,
        n_selected=80,
        selection_rate_per_group={"A": 0.5, "B": 0.3},
    )
    d = asdict(r)
    assert d["round_idx"] == 0
    assert d["metrics"]["accuracy"] == pytest.approx(0.81)
    assert d["selection_rate_per_group"] == {"A": 0.5, "B": 0.3}


# ---------------------------------------------------------------------------
# Simulator behaviour — load-bearing tests
# ---------------------------------------------------------------------------


def test_unbiased_dataset_with_perfect_info_is_approximately_stable():
    """No-bias dataset + perfect-info feedback => DP slope near zero."""
    X, y, s = _unbiased_dataset(n=800, seed=11)
    Xtr, ytr, str_, Xte, yte, ste = _split(X, y, s, seed=11)
    sim = FeedbackLoopSimulator(
        model_factory=_factory,
        sensitive_features=["group"],
        decision_rule=DecisionRule.THRESHOLD,
        feedback_rule=FeedbackRule.PERFECT_INFO,
        n_rounds=8,
        random_state=11,
        threshold=0.5,
    )
    results = sim.run(Xtr, ytr, Xte, yte, str_, ste)
    summary = sim.summary(results)
    # Slope should be small in absolute value. The threshold is generous
    # because LR on a small synthetic set has noise.
    assert abs(summary["dp_slope"]) < 0.01
    assert summary["n_rounds"] == 8


def test_biased_dataset_with_selection_bias_drifts_worse():
    """Biased dataset + selection-bias feedback => DP gets worse over rounds.

    This is the load-bearing test that validates the simulator's premise:
    a stricter selection threshold means group B is initially under-served,
    that under-service feeds back into the next training set, and the gap
    widens.
    """
    X, y, s = _biased_dataset(n=1200, seed=7)
    Xtr, ytr, str_, Xte, yte, ste = _split(X, y, s, seed=7)
    sim = FeedbackLoopSimulator(
        model_factory=_factory,
        sensitive_features=["group"],
        decision_rule=DecisionRule.THRESHOLD,
        feedback_rule=FeedbackRule.SELECTION_BIAS,
        n_rounds=10,
        random_state=7,
        # 0.7 leaves room for the gap to grow; at 0.5 the model already
        # selects nearly everyone in group A on round 0 and there is no
        # headroom to compound.
        threshold=0.7,
    )
    results = sim.run(Xtr, ytr, Xte, yte, str_, ste)
    summary = sim.summary(results)
    # The simulator's premise: bias compounds. DP should be materially
    # worse at the end than at the start.
    assert summary["dp_last"] > summary["dp_first"]
    # ...and the slope should be positive over the run.
    assert summary["dp_slope"] > 0.0


def test_reproducibility_same_seed_same_results():
    X, y, s = _biased_dataset(n=400, seed=3)
    Xtr, ytr, str_, Xte, yte, ste = _split(X, y, s, seed=3)
    common = dict(
        model_factory=_factory,
        sensitive_features=["group"],
        decision_rule=DecisionRule.THRESHOLD,
        feedback_rule=FeedbackRule.SELECTION_BIAS,
        n_rounds=4,
        random_state=99,
        threshold=0.5,
    )
    r1 = FeedbackLoopSimulator(**common).run(Xtr, ytr, Xte, yte, str_, ste)
    r2 = FeedbackLoopSimulator(**common).run(Xtr, ytr, Xte, yte, str_, ste)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.metrics["demographic_parity_difference"] == pytest.approx(
            b.metrics["demographic_parity_difference"]
        )
        assert a.n_train == b.n_train
        assert a.n_selected == b.n_selected


def test_one_round_has_no_drift():
    X, y, s = _biased_dataset(n=400, seed=5)
    Xtr, ytr, str_, Xte, yte, ste = _split(X, y, s, seed=5)
    sim = FeedbackLoopSimulator(
        model_factory=_factory,
        sensitive_features=["group"],
        decision_rule=DecisionRule.THRESHOLD,
        feedback_rule=FeedbackRule.SELECTION_BIAS,
        n_rounds=1,
        random_state=5,
        threshold=0.5,
    )
    results = sim.run(Xtr, ytr, Xte, yte, str_, ste)
    summary = sim.summary(results)
    assert summary["n_rounds"] == 1
    # With one observation, slope is undefined; we report 0.0.
    assert summary["dp_slope"] == 0.0
    assert summary["dp_first"] == summary["dp_last"]


def test_summary_records_drift_headline():
    X, y, s = _biased_dataset(n=300, seed=2)
    Xtr, ytr, str_, Xte, yte, ste = _split(X, y, s, seed=2)
    sim = FeedbackLoopSimulator(
        model_factory=_factory,
        sensitive_features=["group"],
        n_rounds=3,
        random_state=2,
    )
    results = sim.run(Xtr, ytr, Xte, yte, str_, ste)
    s_ = sim.summary(results)
    assert "round 0" in s_["drift_headline"]
    assert "round 2" in s_["drift_headline"]


def test_simulator_rejects_zero_rounds():
    with pytest.raises(ValueError):
        FeedbackLoopSimulator(
            model_factory=_factory,
            sensitive_features=["group"],
            n_rounds=0,
        )


def test_simulator_requires_sensitive_features():
    with pytest.raises(ValueError):
        FeedbackLoopSimulator(
            model_factory=_factory,
            sensitive_features=[],
            n_rounds=2,
        )


def test_per_group_selection_rates_recorded():
    X, y, s = _biased_dataset(n=400, seed=8)
    Xtr, ytr, str_, Xte, yte, ste = _split(X, y, s, seed=8)
    sim = FeedbackLoopSimulator(
        model_factory=_factory,
        sensitive_features=["group"],
        decision_rule=DecisionRule.THRESHOLD,
        feedback_rule=FeedbackRule.SELECTION_BIAS,
        n_rounds=2,
        random_state=8,
    )
    results = sim.run(Xtr, ytr, Xte, yte, str_, ste)
    rates = results[0].selection_rate_per_group
    # Both groups should appear with rates in [0, 1].
    assert set(rates.keys()) >= {"A", "B"}
    for v in rates.values():
        assert 0.0 <= v <= 1.0
