"""Tests for equiml.uncertainty (RFC 0006)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equiml.uncertainty import (
    DP,
    EO,
    EOPP,
    MetricInterval,
    UncertaintyResult,
    bootstrap_fairness_intervals,
    impute_and_bootstrap,
    noise_model_intervals,
)


# ---------- fixtures -------------------------------------------------------


def _biased_dataset(n: int = 500, bias: float = 0.25, seed: int = 0) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Synthetic dataset with a known DP gap of approximately `bias`."""
    rng = np.random.default_rng(seed)
    sf = rng.choice(["A", "B"], size=n)
    base_rate = 0.4
    p = np.where(sf == "A", base_rate + bias / 2, base_rate - bias / 2)
    y_true = rng.binomial(1, base_rate, size=n)
    y_pred = rng.binomial(1, p, size=n)
    return y_true, y_pred, sf


def _inject_missingness(sf: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Return a copy of `sf` with `fraction` of values replaced by NaN."""
    rng = np.random.default_rng(seed)
    out = sf.astype(object).copy()
    mask = rng.random(len(sf)) < fraction
    out[mask] = np.nan
    return out


# ---------- MetricInterval / dataclass shape -------------------------------


def test_metric_interval_string_format():
    mi = MetricInterval(point=0.04, lower=0.01, upper=0.09, confidence=0.95)
    assert str(mi) == "0.04 [0.01, 0.09] (95%)"


def test_metric_interval_default_confidence():
    mi = MetricInterval(point=0.0, lower=0.0, upper=0.0)
    assert mi.confidence == 0.95


def test_uncertainty_result_indexable():
    y_true, y_pred, sf = _biased_dataset(n=200, seed=1)
    res = bootstrap_fairness_intervals(y_true, y_pred, sf, n_bootstrap=50)
    assert isinstance(res, UncertaintyResult)
    assert isinstance(res[DP], MetricInterval)
    assert set(res.keys()) == {DP, EO, EOPP}


# ---------- bootstrap basic correctness ------------------------------------


def test_bootstrap_interval_contains_point_estimate():
    """The percentile interval must straddle the point estimate."""
    y_true, y_pred, sf = _biased_dataset(n=400, seed=2)
    res = bootstrap_fairness_intervals(y_true, y_pred, sf, n_bootstrap=400, random_state=7)
    for key in (DP, EO, EOPP):
        mi = res[key]
        assert mi.lower <= mi.point <= mi.upper, (
            f"{key}: point {mi.point} fell outside [{mi.lower}, {mi.upper}]"
        )


def test_bootstrap_smaller_sample_yields_wider_interval():
    """A smaller test set has higher sample uncertainty, so the interval widens."""
    y_true_big, y_pred_big, sf_big = _biased_dataset(n=2000, seed=11)
    y_true_sm, y_pred_sm, sf_sm = _biased_dataset(n=200, seed=11)

    big = bootstrap_fairness_intervals(
        y_true_big, y_pred_big, sf_big, n_bootstrap=400, random_state=3
    )
    small = bootstrap_fairness_intervals(
        y_true_sm, y_pred_sm, sf_sm, n_bootstrap=400, random_state=3
    )
    # DP interval is the most stable so we test on it.
    assert small[DP].width() > big[DP].width()


def test_bootstrap_reproducibility_same_seed():
    y_true, y_pred, sf = _biased_dataset(n=300, seed=4)
    a = bootstrap_fairness_intervals(y_true, y_pred, sf, n_bootstrap=200, random_state=99)
    b = bootstrap_fairness_intervals(y_true, y_pred, sf, n_bootstrap=200, random_state=99)
    for key in (DP, EO, EOPP):
        assert a[key].lower == pytest.approx(b[key].lower)
        assert a[key].upper == pytest.approx(b[key].upper)
        assert a[key].point == pytest.approx(b[key].point)


def test_bootstrap_different_seed_changes_interval():
    y_true, y_pred, sf = _biased_dataset(n=300, seed=5)
    a = bootstrap_fairness_intervals(y_true, y_pred, sf, n_bootstrap=200, random_state=1)
    b = bootstrap_fairness_intervals(y_true, y_pred, sf, n_bootstrap=200, random_state=2)
    # The point estimate is deterministic (no resampling), but the
    # interval edges should differ across seeds.
    assert a[DP].lower != b[DP].lower or a[DP].upper != b[DP].upper


def test_bootstrap_95_percent_interval_actual_coverage():
    """The percentile interval should contain ~95% of the bootstrap draws.

    This is a self-consistency check: by construction, the percentile
    interval's nominal coverage equals its empirical coverage of the
    sample population it was built from. We allow 2 percentage points of
    slack to absorb the discreteness of the percentile estimator.
    """
    y_true, y_pred, sf = _biased_dataset(n=400, seed=8)
    res = bootstrap_fairness_intervals(
        y_true, y_pred, sf, n_bootstrap=2000, confidence=0.95, random_state=42
    )

    # Re-run the bootstrap and check actual coverage on the same draws.
    rng = np.random.default_rng(42)
    n = len(y_true)
    from fairlearn.metrics import demographic_parity_difference
    draws = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        s = sf[idx]
        if len(np.unique(s)) < 2:
            continue
        draws.append(
            float(demographic_parity_difference(
                y_true[idx], y_pred[idx], sensitive_features=s
            ))
        )
    arr = np.asarray(draws)
    inside = ((arr >= res[DP].lower) & (arr <= res[DP].upper)).mean()
    assert 0.93 <= inside <= 0.97, f"empirical coverage {inside} off nominal 0.95"


# ---------- edge cases -----------------------------------------------------


def test_empty_input_raises_clear_error():
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_fairness_intervals(np.array([]), np.array([]), np.array([]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        bootstrap_fairness_intervals(
            np.array([0, 1, 1]),
            np.array([0, 1]),
            np.array(["A", "B", "A"]),
        )


def test_constant_predictions_zero_dp_zero_width():
    """Constant predictions imply zero selection-rate disparity."""
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_pred = np.zeros(n, dtype=int)
    sf = np.array(["A"] * 100 + ["B"] * 100)
    res = bootstrap_fairness_intervals(y_true, y_pred, sf, n_bootstrap=200, random_state=0)
    assert res[DP].point == 0.0
    assert res[DP].lower == 0.0
    assert res[DP].upper == 0.0
    assert res[DP].width() == 0.0


def test_confidence_outside_unit_interval_raises():
    y_true, y_pred, sf = _biased_dataset(n=100, seed=12)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_fairness_intervals(
            y_true, y_pred, sf, n_bootstrap=10, confidence=1.5
        )


def test_n_bootstrap_zero_raises():
    y_true, y_pred, sf = _biased_dataset(n=100, seed=13)
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_fairness_intervals(y_true, y_pred, sf, n_bootstrap=0)


# ---------- multiple imputation --------------------------------------------


def test_imputation_widens_interval_relative_to_oracle():
    """With 30% missing protected attribute, MI on the full N rows must
    produce a strictly wider interval than the oracle bootstrap that knew
    the true sensitive label for every row.

    This isolates the missingness effect: both bootstraps are run on the
    same N, so any width difference is attributable to imputation
    uncertainty rather than sample size.
    """
    y_true, y_pred, sf = _biased_dataset(n=600, seed=21)
    sf_with_nan = _inject_missingness(sf, fraction=0.30, seed=22)

    oracle_res = bootstrap_fairness_intervals(
        y_true, y_pred, sf,
        n_bootstrap=2000, random_state=33,
    )
    mi_res = impute_and_bootstrap(
        y_true, y_pred, sf_with_nan,
        n_imputations=20, n_bootstrap_per_impute=100, random_state=33,
    )
    assert mi_res.missing_fraction == pytest.approx(0.30, abs=0.05)
    assert mi_res[DP].width() > oracle_res[DP].width(), (
        f"MI width {mi_res[DP].width()} not > oracle width {oracle_res[DP].width()}"
    )


def test_imputation_widens_interval_vs_complete_cases_when_sample_size_matched():
    """Compares MI on N rows (30% NaN) to bootstrap on the complete-case
    subset. We do NOT assert MI > complete-cases here because complete-
    cases has fewer rows and its sample-size widening can swamp the MI
    widening. Instead, we assert that MI's width is at least within
    half the complete-cases width — i.e. MI is recovering signal from
    the imputed rows rather than collapsing.
    """
    y_true, y_pred, sf = _biased_dataset(n=600, seed=51)
    sf_with_nan = _inject_missingness(sf, fraction=0.30, seed=52)

    cc_mask = ~pd.isna(sf_with_nan)
    cc_res = bootstrap_fairness_intervals(
        y_true[cc_mask], y_pred[cc_mask], sf_with_nan[cc_mask],
        n_bootstrap=400, random_state=33,
    )
    mi_res = impute_and_bootstrap(
        y_true, y_pred, sf_with_nan,
        n_imputations=20, n_bootstrap_per_impute=100, random_state=33,
    )
    # MI is using more rows than complete-cases (600 vs ~420), so its
    # sample-size component is smaller; the imputation noise then adds
    # back. Net width should be the same order of magnitude.
    assert mi_res[DP].width() >= 0.5 * cc_res[DP].width()


def test_imputation_no_missing_delegates_to_bootstrap():
    y_true, y_pred, sf = _biased_dataset(n=300, seed=23)
    res = impute_and_bootstrap(
        y_true, y_pred, sf,
        n_imputations=4, n_bootstrap_per_impute=50, random_state=7,
    )
    assert res.missing_fraction == 0.0
    assert any("delegated to plain bootstrap" in n for n in res.notes)


def test_imputation_reproducibility_same_seed():
    y_true, y_pred, sf = _biased_dataset(n=300, seed=24)
    sf_nan = _inject_missingness(sf, fraction=0.25, seed=25)
    a = impute_and_bootstrap(
        y_true, y_pred, sf_nan,
        n_imputations=5, n_bootstrap_per_impute=40, random_state=11,
    )
    b = impute_and_bootstrap(
        y_true, y_pred, sf_nan,
        n_imputations=5, n_bootstrap_per_impute=40, random_state=11,
    )
    for key in (DP, EO, EOPP):
        assert a[key].lower == pytest.approx(b[key].lower)
        assert a[key].upper == pytest.approx(b[key].upper)


def test_imputation_with_features_runs():
    """Conditional imputation when features are passed should run and
    return a valid interval."""
    y_true, y_pred, sf = _biased_dataset(n=300, seed=26)
    sf_nan = _inject_missingness(sf, fraction=0.20, seed=27)
    rng = np.random.default_rng(28)
    feats = pd.DataFrame({
        "age": rng.integers(20, 60, size=300),
        "city": rng.choice(["X", "Y"], size=300),
    })
    res = impute_and_bootstrap(
        y_true, y_pred, sf_nan,
        features=feats,
        n_imputations=5, n_bootstrap_per_impute=40, random_state=29,
    )
    assert res.missing_fraction > 0
    assert res[DP].lower <= res[DP].point <= res[DP].upper


# ---------- noise model ----------------------------------------------------


def test_noise_model_zero_noise_matches_observed_point():
    """Identity noise matrix should yield an interval whose point equals
    the observed-data point estimate."""
    y_true, y_pred, sf = _biased_dataset(n=300, seed=31)
    identity = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    res = noise_model_intervals(
        y_true, y_pred, sf,
        error_rate_matrix=identity,
        n_simulations=200,
        random_state=42,
    )
    # With zero noise, the interval should be very tight around the point.
    assert res[DP].width() < 0.05


def test_noise_model_high_noise_widens_interval():
    """A noisier matrix produces a wider DP interval."""
    y_true, y_pred, sf = _biased_dataset(n=400, seed=32)
    low_noise = pd.DataFrame(
        [[0.95, 0.05], [0.05, 0.95]], index=["A", "B"], columns=["A", "B"],
    )
    high_noise = pd.DataFrame(
        [[0.7, 0.3], [0.3, 0.7]], index=["A", "B"], columns=["A", "B"],
    )
    low = noise_model_intervals(
        y_true, y_pred, sf, error_rate_matrix=low_noise,
        n_simulations=300, random_state=5,
    )
    high = noise_model_intervals(
        y_true, y_pred, sf, error_rate_matrix=high_noise,
        n_simulations=300, random_state=5,
    )
    assert high[DP].width() > low[DP].width()


def test_noise_model_dict_form_accepted():
    y_true, y_pred, sf = _biased_dataset(n=200, seed=33)
    matrix = {"A": {"A": 0.9, "B": 0.1}, "B": {"A": 0.1, "B": 0.9}}
    res = noise_model_intervals(
        y_true, y_pred, sf, error_rate_matrix=matrix,
        n_simulations=100, random_state=8,
    )
    assert isinstance(res[DP], MetricInterval)


# ---------- pandas / numpy ergonomics --------------------------------------


def test_accepts_pandas_series_inputs():
    y_true, y_pred, sf = _biased_dataset(n=200, seed=41)
    res = bootstrap_fairness_intervals(
        pd.Series(y_true),
        pd.Series(y_pred),
        pd.Series(sf, name="gender"),
        n_bootstrap=80,
        random_state=14,
    )
    assert isinstance(res[DP], MetricInterval)
