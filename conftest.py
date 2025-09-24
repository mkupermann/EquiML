"""
Pytest configuration for EquiML
Ensures proper test environment setup and import paths
"""

import sys
import os
import pytest

# Add src directory to Python path for all tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def pytest_configure(config):
    """Configure pytest environment"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment for all tests"""

    # Ensure test data directory exists
    test_data_dir = os.path.join(os.path.dirname(__file__), 'tests', 'data')
    os.makedirs(test_data_dir, exist_ok=True)

    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['PYTEST_RUNNING'] = 'true'

    print("✅ Test environment configured")

@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame for testing"""
    try:
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'category': np.random.choice(['A', 'B'], 100),
            'sensitive': np.random.choice(['Group1', 'Group2'], 100),
            'target': np.random.choice([0, 1], 100)
        })
    except ImportError:
        pytest.skip("pandas/numpy not available")