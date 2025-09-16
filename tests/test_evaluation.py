import pandas as pd
import numpy as np
from src.model import Model
from src.evaluation import EquiMLEvaluation
from src.data import Data

def test_fairness_metrics():
    # Create synthetic data
    data = Data(sensitive_features=['sensitive'])
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [5, 6, 7, 8, 9, 10],
        'sensitive': ['A', 'A', 'A', 'B', 'B', 'B'],
        'target': [0, 0, 1, 0, 1, 1]
    })
    data.df = df
    data.preprocess(target_column='target', categorical_features=['sensitive'])
    data.split_data(test_size=0.5, random_state=42)

    # Train a model
    model = Model(algorithm='logistic_regression')
    model.train(data.X_train, data.y_train)

    # Evaluate fairness
    evaluation = EquiMLEvaluation()
    metrics = evaluation.evaluate(model, data.X_test, data.y_test, sensitive_features=data.X_test['sensitive_B'])

    assert 'demographic_parity_difference' in metrics
    assert 'equalized_odds_difference' in metrics
    assert 'disparate_impact' in metrics['fairness_metrics']

# Run the test
test_fairness_metrics()
