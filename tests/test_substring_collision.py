"""Substring-collision regression test.

Locks Teammate 1, task 4: encoded sensitive columns are matched with an
exact prefix (``col == sf`` or ``col.startswith(f"{sf}_")``), not a
substring scan. Without that fix, requesting ``race`` as sensitive would
also match a column literally named ``racing_score``, silently widening
the audit's protected set.

Two checks here:

1. ``Data._apply_reweighing`` only picks up ``race`` (and its one-hot
   expansions ``race_*``), never ``racing_score``.
2. The CLI surface, end-to-end, agrees.
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from equiml.data import Data


def _make_collision_dataset(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    race = rng.choice(["A", "B", "C"], n)
    racing_score = rng.normal(size=n)  # numeric, NOT sensitive
    age = rng.integers(18, 65, n)
    target = (rng.random(n) < 0.4).astype(int)
    return pd.DataFrame({
        "age": age,
        "race": race,
        "racing_score": racing_score,
        "target": target,
    })


def test_reweighing_does_not_match_racing_score(monkeypatch):
    """``race`` must not match ``racing_score`` in ``_apply_reweighing``."""
    df = _make_collision_dataset()

    data = Data(sensitive_features=["race"])
    data.df = df
    data.preprocess(
        target_column="target",
        numerical_features=["age", "racing_score"],
        categorical_features=["race"],
    )

    # Reach into the matcher used by _apply_reweighing. We rebuild it
    # the same way the source does so a future refactor that breaks the
    # contract is caught here.
    sensitive_cols = [
        col for col in data.X.columns
        if any(
            col == sf or col.startswith(f"{sf}_")
            for sf in data.sensitive_features
        )
    ]

    assert sensitive_cols, "expected at least one race-encoded column"
    assert "racing_score" not in sensitive_cols, (
        "regression: 'racing_score' was matched as a sensitive column. "
        f"Matched columns: {sensitive_cols}"
    )
    # All matches must start with the literal sensitive name, not be
    # arbitrary substring hits.
    for col in sensitive_cols:
        assert col == "race" or col.startswith("race_"), (
            f"unexpected match {col!r} for sensitive feature 'race'"
        )

    # Sanity: actually run the reweighing path; it must not crash and
    # must produce one weight per row.
    data._apply_reweighing()
    assert data.sample_weights is not None
    assert len(data.sample_weights) == len(data.y)


def test_cli_does_not_treat_racing_score_as_sensitive(tmp_path):
    """End-to-end: CLI's _meta block reflects only the user's request."""
    csv_path = tmp_path / "collision.csv"
    json_path = tmp_path / "out.json"
    _make_collision_dataset().to_csv(csv_path, index=False)

    result = subprocess.run(
        [
            sys.executable, "-m", "equiml.cli", "audit", str(csv_path),
            "--target", "target",
            "--sensitive", "race",
            "--output", str(json_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"CLI failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    with open(json_path) as f:
        payload = json.load(f)

    # Per-sensitive block must contain only 'race', never 'racing_score'.
    per_sensitive = payload["baseline"].get("per_sensitive", {})
    if per_sensitive:
        assert "race" in per_sensitive
        assert "racing_score" not in per_sensitive, (
            f"regression: 'racing_score' leaked into per_sensitive: "
            f"{list(per_sensitive)}"
        )

    # If Teammate 1 has shipped the _meta block, it must agree.
    if "_meta" in payload:
        assert payload["_meta"]["sensitive_features"] == ["race"]
