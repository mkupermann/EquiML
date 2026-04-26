"""End-to-end CLI smoke test.

Invokes ``equiml audit`` as a real subprocess against a synthetic CSV and
verifies the audit produces the contracts the README promises:

* exit code 0,
* a JSON file with ``baseline``, ``fair``, and ``_meta`` blocks,
* an HTML report file that is non-empty,
* ``_meta.equiml_version`` matching ``equiml.__version__`` (contract from
  Teammate 1, task 7).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import equiml


def _write_synthetic_csv(path) -> None:
    """Write a 300-row adult-style CSV at ``path``."""
    rng = np.random.default_rng(42)
    n = 300
    gender = rng.choice(["Male", "Female"], n)
    age = rng.integers(18, 65, n)
    hours = rng.integers(10, 60, n)
    income = np.where(
        (gender == "Male") & (hours > 35),
        rng.choice([0, 1], n, p=[0.3, 0.7]),
        rng.choice([0, 1], n, p=[0.7, 0.3]),
    )
    pd.DataFrame({
        "age": age,
        "hours_per_week": hours,
        "gender": gender,
        "income": income,
    }).to_csv(path, index=False)


def _run_cli(args, cwd=None):
    """Run the equiml CLI as a subprocess via ``python -m equiml.cli``.

    Using the module form (instead of the ``equiml`` console-script) means
    the test is portable across CI environments where the entry point may
    not be on ``$PATH`` yet.
    """
    cmd = [sys.executable, "-m", "equiml.cli"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def test_cli_audit_smoke(tmp_path):
    csv_path = tmp_path / "adult.csv"
    json_path = tmp_path / "results.json"
    html_path = tmp_path / "report.html"
    _write_synthetic_csv(csv_path)

    result = _run_cli([
        "audit",
        str(csv_path),
        "--target", "income",
        "--sensitive", "gender",
        "--output", str(json_path),
        "--report", str(html_path),
    ])

    assert result.returncode == 0, (
        f"CLI exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # JSON output exists, parses, has the documented top-level structure.
    assert json_path.exists(), "JSON output file was not created"
    with open(json_path) as f:
        payload = json.load(f)

    assert "baseline" in payload, f"missing 'baseline' key; got {list(payload)}"
    assert "fair" in payload, f"missing 'fair' key; got {list(payload)}"
    assert "_meta" in payload, (
        f"missing '_meta' block (Teammate 1 task 7); got {list(payload)}"
    )

    # _meta block locks the version-stamp contract.
    meta = payload["_meta"]
    assert meta["equiml_version"] == equiml.__version__, (
        f"_meta.equiml_version={meta.get('equiml_version')!r} does not match "
        f"equiml.__version__={equiml.__version__!r}"
    )
    for required in (
        "python_version",
        "sklearn_version",
        "fairlearn_version",
        "random_seed",
        "dataset_path",
        "target",
        "sensitive_features",
        "algorithm",
    ):
        assert required in meta, f"_meta missing required key {required!r}"

    assert meta["target"] == "income"
    assert meta["sensitive_features"] == ["gender"]

    # HTML report exists and is non-trivial.
    assert html_path.exists(), "HTML report was not created"
    assert html_path.stat().st_size > 0, "HTML report is empty"


def test_cli_audit_missing_file(tmp_path):
    """Missing dataset path must fail loudly, not silently."""
    result = _run_cli([
        "audit",
        str(tmp_path / "does_not_exist.csv"),
        "--target", "income",
        "--sensitive", "gender",
    ])
    assert result.returncode != 0
