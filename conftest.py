"""Pytest configuration for EquiML."""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    os.environ["TESTING"] = "true"


@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame for testing."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    return pd.DataFrame({
        "feature1": np.random.randn(200),
        "feature2": np.random.randn(200),
        "category": np.random.choice(["A", "B"], 200),
        "sensitive": np.random.choice(["Group1", "Group2"], 200),
        "target": np.random.choice([0, 1], 200),
    })


@pytest.fixture
def adult_sample():
    """Minimal synthetic dataset mimicking the Adult Census structure.

    Shared across the test suite (see tests/test_audit_pipeline.py and
    tests/test_cli.py). Bias is injected on `gender` so fairness metrics
    actually move when the audit runs.
    """
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 300
    gender = np.random.choice(["Male", "Female"], n)
    age = np.random.randint(18, 65, n)
    hours = np.random.randint(10, 60, n)
    # Introduce bias: males more likely to have high income
    income = np.where(
        (gender == "Male") & (hours > 35),
        np.random.choice([0, 1], n, p=[0.3, 0.7]),
        np.random.choice([0, 1], n, p=[0.7, 0.3]),
    )
    return pd.DataFrame({
        "age": age,
        "hours_per_week": hours,
        "gender": gender,
        "income": income,
    })
