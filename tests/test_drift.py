"""Tests for FairnessDriftMonitor (RFC 0003)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equiml.drift import (
    PSI_MATERIAL_DRIFT,
    PSI_NO_DRIFT,
    Batch,
    FairnessDriftMonitor,
    _psi,
    render_drift_report,
)


# --- PSI sanity ------------------------------------------------------------

def test_psi_zero_for_identical_distributions():
    a = {"male": 0.5, "female": 0.5}
    assert _psi(a, a) == pytest.approx(0.0, abs=1e-6)


def test_psi_grows_with_distribution_shift():
    base = {"male": 0.5, "female": 0.5}
    small = {"male": 0.55, "female": 0.45}
    big = {"male": 0.9, "female": 0.1}
    assert _psi(base, small) < _psi(base, big)


def test_psi_threshold_bands_are_credit_risk_convention():
    # Sanity: the constants we expose match the documented bands.
    assert PSI_NO_DRIFT == 0.10
    assert PSI_MATERIAL_DRIFT == 0.25


# --- Persistence / round-trip ----------------------------------------------

def _df(n, rng_seed, base_rate_male=0.5, base_rate_female=0.3):
    rng = np.random.default_rng(rng_seed)
    sex = rng.choice(["male", "female"], n)
    preds = np.where(
        sex == "male",
        rng.binomial(1, base_rate_male, n),
        rng.binomial(1, base_rate_female, n),
    )
    return pd.DataFrame({"sex": sex}), preds


def test_record_then_load_round_trip(tmp_path):
    state = tmp_path / "monitor.jsonl"
    sf, preds = _df(500, rng_seed=1)

    m1 = FairnessDriftMonitor(["sex"], state_path=state)
    batch1 = m1.record(predictions=preds, sensitive_features=sf, metadata={"v": "1.0"})
    assert state.exists()

    # Reload from disk
    m2 = FairnessDriftMonitor(["sex"], state_path=state)
    assert len(m2.batches) == 1
    assert m2.batches[0].n_samples == 500
    assert m2.batches[0].metadata == {"v": "1.0"}
    # Group-rate keys preserved.
    assert set(m2.batches[0].group_rates["sex"]) == {"male", "female"}


def test_record_append_does_not_duplicate(tmp_path):
    state = tmp_path / "monitor.jsonl"
    sf, preds = _df(200, rng_seed=2)

    m1 = FairnessDriftMonitor(["sex"], state_path=state)
    m1.record(predictions=preds, sensitive_features=sf)
    m1.record(predictions=preds, sensitive_features=sf)
    # State file should have exactly two lines.
    lines = [line for line in state.read_text().splitlines() if line.strip()]
    assert len(lines) == 2

    m2 = FairnessDriftMonitor(["sex"], state_path=state)
    assert len(m2.batches) == 2


def test_record_requires_dataframe_for_sensitive():
    m = FairnessDriftMonitor(["sex"])
    sf, preds = _df(50, rng_seed=3)
    with pytest.raises(TypeError, match="DataFrame"):
        m.record(predictions=preds, sensitive_features=sf["sex"])  # Series, not DataFrame


# --- Window selection ------------------------------------------------------

def test_window_by_days_and_n_batches():
    m = FairnessDriftMonitor(["sex"])
    now = datetime.now(timezone.utc)
    for i in range(20):
        ts = now - timedelta(days=20 - i)
        sf, preds = _df(100, rng_seed=10 + i)
        m.record(predictions=preds, sensitive_features=sf, timestamp=ts)
    # Last 5 days (inclusive of today)
    last_5_days = m.window(days=5)
    assert 4 <= len(last_5_days) <= 6  # account for date boundary
    # Last 3 batches
    last_3 = m.window(n_batches=3)
    assert len(last_3) == 3


# --- Drift detection -------------------------------------------------------

def test_no_drift_when_distributions_match():
    m = FairnessDriftMonitor(["sex"])
    now = datetime.now(timezone.utc)
    for i in range(10):
        sf, preds = _df(500, rng_seed=100 + i, base_rate_male=0.5, base_rate_female=0.4)
        m.record(predictions=preds, sensitive_features=sf,
                 timestamp=now - timedelta(days=20 - i))
    for i in range(5):
        sf, preds = _df(500, rng_seed=200 + i, base_rate_male=0.5, base_rate_female=0.4)
        m.record(predictions=preds, sensitive_features=sf,
                 timestamp=now - timedelta(days=5 - i))
    drift = m.detect_drift(
        current_window=m.window(days=5),
        baseline_window=m.window(days=15, until=now - timedelta(days=5)),
    )
    assert drift.severity == "none"
    assert drift.psi_overall < PSI_NO_DRIFT


def test_drift_detected_when_distributions_diverge():
    m = FairnessDriftMonitor(["sex"])
    now = datetime.now(timezone.utc)
    # Baseline: male 0.5, female 0.4
    for i in range(10):
        sf, preds = _df(500, rng_seed=300 + i, base_rate_male=0.5, base_rate_female=0.4)
        m.record(predictions=preds, sensitive_features=sf,
                 timestamp=now - timedelta(days=20 - i))
    # Current: male 0.9, female 0.1 — strong divergence
    for i in range(5):
        sf, preds = _df(500, rng_seed=400 + i, base_rate_male=0.9, base_rate_female=0.1)
        m.record(predictions=preds, sensitive_features=sf,
                 timestamp=now - timedelta(days=5 - i))
    drift = m.detect_drift(
        current_window=m.window(days=5),
        baseline_window=m.window(days=15, until=now - timedelta(days=5)),
    )
    assert drift.severity in {"moderate", "material"}
    assert drift.psi_overall > 0.1


def test_drift_handles_empty_baseline_gracefully():
    m = FairnessDriftMonitor(["sex"])
    sf, preds = _df(100, rng_seed=999)
    m.record(predictions=preds, sensitive_features=sf)
    drift = m.detect_drift(current_window=m.window(), baseline_window=[])
    assert drift.severity == "none"
    assert any("baseline window empty" in n for n in drift.notes)


# --- Window aggregation -----------------------------------------------------

def test_aggregate_window_weights_by_n_samples():
    m = FairnessDriftMonitor(["sex"])
    sf1, preds1 = _df(100, rng_seed=1, base_rate_male=0.8, base_rate_female=0.2)
    sf2, preds2 = _df(900, rng_seed=2, base_rate_male=0.4, base_rate_female=0.4)
    m.record(predictions=preds1, sensitive_features=sf1)
    m.record(predictions=preds2, sensitive_features=sf2)
    agg = m.aggregate_window(m.window())
    # The weighted-average male rate should be much closer to 0.4 (the larger
    # batch) than to 0.8 (the smaller).
    assert agg["group_rates"]["sex"]["male"] < 0.55


def test_aggregate_empty_window():
    m = FairnessDriftMonitor(["sex"])
    agg = m.aggregate_window([])
    assert agg["n_samples"] == 0
    assert agg["metrics"] == {}


# --- Policy integration ----------------------------------------------------

def test_evaluate_against_policy_uses_aggregated_window(tmp_path):
    from equiml.policy import load_policy

    m = FairnessDriftMonitor(["sex"])
    sf, preds = _df(1000, rng_seed=7, base_rate_male=0.8, base_rate_female=0.2)
    m.record(predictions=preds, sensitive_features=sf)

    policy_path = tmp_path / "fairness.yaml"
    policy_path.write_text("""
version: 1
target: outcome
sensitive: [sex]
gates:
  demographic_parity_difference: { max: 0.05 }
""")
    policy = load_policy(policy_path)
    result = m.evaluate_against_policy(policy)
    # The 0.8 vs 0.2 setup is way over the 0.05 max.
    assert result.passed is False


# --- CLI integration via subprocess ----------------------------------------

@pytest.fixture
def fixture_batch_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n = 500
    sex = rng.choice(["male", "female"], n)
    preds = np.where(sex == "male", rng.binomial(1, 0.55, n), rng.binomial(1, 0.45, n))
    df = pd.DataFrame({"prediction": preds, "sex": sex, "actual": preds})
    p = tmp_path / "batch.csv"
    df.to_csv(p, index=False)
    return p


def test_cli_monitor_record_creates_state(tmp_path, fixture_batch_csv):
    state = tmp_path / "monitor.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "monitor", "record",
         "--state", str(state),
         "--batch", str(fixture_batch_csv),
         "--predictions-col", "prediction",
         "--sensitive", "sex"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state.exists()
    line = state.read_text().splitlines()[0]
    payload = json.loads(line)
    assert payload["n_samples"] == 500


def test_cli_monitor_check_returns_0_when_no_drift_and_no_policy(tmp_path, fixture_batch_csv):
    state = tmp_path / "monitor.jsonl"
    # Record several batches with the same distribution.
    for _ in range(5):
        subprocess.run(
            [sys.executable, "-m", "equiml.cli", "monitor", "record",
             "--state", str(state),
             "--batch", str(fixture_batch_csv),
             "--predictions-col", "prediction",
             "--sensitive", "sex"],
            check=True, capture_output=True,
        )
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "monitor", "check",
         "--state", str(state),
         "--sensitive", "sex",
         "--baseline-days", "30",
         "--current-days", "7"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_monitor_check_returns_3_on_policy_breach(tmp_path, fixture_batch_csv):
    state = tmp_path / "monitor.jsonl"
    subprocess.run(
        [sys.executable, "-m", "equiml.cli", "monitor", "record",
         "--state", str(state),
         "--batch", str(fixture_batch_csv),
         "--predictions-col", "prediction",
         "--sensitive", "sex"],
        check=True, capture_output=True,
    )
    policy = tmp_path / "fairness.yaml"
    policy.write_text("""
version: 1
target: outcome
sensitive: [sex]
gates:
  demographic_parity_difference: { max: 0.001 }
""")
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "monitor", "check",
         "--state", str(state),
         "--sensitive", "sex",
         "--policy", str(policy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 3, result.stdout + result.stderr


def test_cli_monitor_report_writes_markdown(tmp_path, fixture_batch_csv):
    state = tmp_path / "monitor.jsonl"
    for _ in range(3):
        subprocess.run(
            [sys.executable, "-m", "equiml.cli", "monitor", "record",
             "--state", str(state),
             "--batch", str(fixture_batch_csv),
             "--predictions-col", "prediction",
             "--sensitive", "sex"],
            check=True, capture_output=True,
        )
    out = tmp_path / "drift_report.md"
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "monitor", "report",
         "--state", str(state),
         "--sensitive", "sex",
         "--output", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.exists()
    text = out.read_text()
    assert "Fairness drift report" in text
    assert "PSI" in text


# --- Render report direct API ---------------------------------------------

def test_render_drift_report_runs_on_minimal_monitor(tmp_path):
    m = FairnessDriftMonitor(["sex"])
    now = datetime.now(timezone.utc)
    for i in range(5):
        sf, preds = _df(100, rng_seed=20 + i)
        m.record(predictions=preds, sensitive_features=sf,
                 timestamp=now - timedelta(days=10 - i))
    drift = m.detect_drift(
        current_window=m.window(n_batches=2),
        baseline_window=m.window(n_batches=3),
    )
    md = render_drift_report(
        monitor=m,
        baseline_window=m.window(n_batches=3),
        current_window=m.window(n_batches=2),
        drift=drift,
    )
    assert "Fairness drift report" in md
    assert "PSI" in md
