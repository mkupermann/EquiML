import pandas as pd
import numpy as np
from src.model import Model
from src.data import Data
import pytest

@pytest.fixture
def dummy_data():
    """Provides a dummy dataset for model testing."""
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'sensitive': np.random.choice(['A', 'B'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    data = Data(sensitive_features=['sensitive'])
    data.df = df
    data.preprocess(target_column='target', numerical_features=['feature1', 'feature2'], categorical_features=['sensitive'])
    data.split_data()
    return data

def test_train_with_sample_weights(dummy_data):
    """Tests that the model can be trained with sample weights."""
    model = Model(algorithm='logistic_regression')
    sample_weight = np.random.rand(len(dummy_data.y_train))
    model.train(dummy_data.X_train, dummy_data.y_train, sample_weight=sample_weight)
    assert hasattr(model.model, 'coef_') # Check if model is fitted

def test_xgboost_algorithm(dummy_data):
    """Tests the XGBoost algorithm."""
    model = Model(algorithm='xgboost')
    model.train(dummy_data.X_train, dummy_data.y_train)
    predictions = model.predict(dummy_data.X_test)
    assert len(predictions) == len(dummy_data.y_test)

def test_tune_hyperparameters(dummy_data):
    """Tests the tune_hyperparameters method."""
    model = Model(algorithm='logistic_regression')
    best_params = model.tune_hyperparameters(dummy_data.X_train, dummy_data.y_train, n_trials=5)
    assert isinstance(best_params, dict)
    assert 'C' in best_params
    assert model.model.get_params()['C'] == best_params['C']

def test_lightgbm_algorithm(dummy_data):
    """Tests the LightGBM algorithm."""
    model = Model(algorithm='lightgbm')
    model.train(dummy_data.X_train, dummy_data.y_train)
    predictions = model.predict(dummy_data.X_test)
    assert len(predictions) == len(dummy_data.y_test)
