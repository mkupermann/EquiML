"""Multi-sensitive contract test.

Locks Teammate 1, task 3: the README's headline example is
``--sensitive gender race``. The audit must produce per-feature metrics
under ``metrics["per_sensitive"][<name>]`` for every sensitive column,
not silently drop all but the first.

The fairlearn constraint training itself still runs once on the *primary*
(first) sensitive feature — that's a documented limitation of
``ExponentiatedGradient``, not what this test is about. This test is
about the audit-side coverage.
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd


def _make_two_sensitive_csv(path) -> None:
    """Build a 400-row dataset with both ``gender`` and ``race`` as sensitive."""
    rng = np.random.default_rng(7)
    n = 400
    gender = rng.choice(["Male", "Female"], n)
    race = rng.choice(["A", "B", "C"], n)
    age = rng.integers(18, 65, n)
    hours = rng.integers(10, 60, n)
    # Bias on gender; mild bias on race.
    gender_boost = (gender == "Male").astype(int)
    race_boost = (race == "A").astype(int)
    base_p = 0.25 + 0.35 * gender_boost + 0.15 * race_boost
    income = (rng.random(n) < base_p).astype(int)
    pd.DataFrame({
        "age": age,
        "hours_per_week": hours,
        "gender": gender,
        "race": race,
        "income": income,
    }).to_csv(path, index=False)


def test_audit_emits_per_sensitive_block(tmp_path):
    csv_path = tmp_path / "two_sensitive.csv"
    json_path = tmp_path / "results.json"
    _make_two_sensitive_csv(csv_path)

    cmd = [
        sys.executable, "-m", "equiml.cli", "audit", str(csv_path),
        "--target", "income",
        "--sensitive", "gender", "race",
        "--output", str(json_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"CLI failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    with open(json_path) as f:
        payload = json.load(f)

    baseline = payload["baseline"]
    assert "per_sensitive" in baseline, (
        f"baseline missing 'per_sensitive' block; got keys {list(baseline)}"
    )
    per_sensitive = baseline["per_sensitive"]

    # Both sensitive columns must produce their own metric block.
    assert "gender" in per_sensitive, (
        f"per_sensitive missing 'gender'; got {list(per_sensitive)}"
    )
    assert "race" in per_sensitive, (
        f"per_sensitive missing 'race'; got {list(per_sensitive)}"
    )

    # Each per-sensitive block has the metrics that downstream code reads.
    for sf in ("gender", "race"):
        block = per_sensitive[sf]
        assert "demographic_parity_difference" in block, (
            f"per_sensitive[{sf!r}] missing demographic_parity_difference; "
            f"got keys {list(block)}"
        )
        # Value must be a real number, not None / NaN sentinel.
        dpd = block["demographic_parity_difference"]
        assert isinstance(dpd, (int, float)), (
            f"per_sensitive[{sf!r}].demographic_parity_difference is "
            f"{dpd!r}, expected a number"
        )

    # Same contract on the fair-model side.
    fair_per_sensitive = payload["fair"].get("per_sensitive", {})
    assert "gender" in fair_per_sensitive
    assert "race" in fair_per_sensitive
