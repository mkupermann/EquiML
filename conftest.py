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
