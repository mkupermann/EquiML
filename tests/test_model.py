import pandas as pd
from src.model import Model
from src.data import Data

def test_model_training_and_evaluation():
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
    
    # Test logistic regression without fairness
    model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')
    model.train(data.X_train, data.y_train, sensitive_features=data.X_train['sensitive_B'])
    metrics = model.evaluate(data.X_test, data.y_test, sensitive_features=data.X_test['sensitive_B'])
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    assert 'demographic_parity_difference' in metrics
    
    # Test logistic regression with fairness constraint
    model_fair = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')
    model_fair.train(data.X_train, data.y_train, sensitive_features=data.X_train[['sensitive_B']])
    metrics_fair = model_fair.evaluate(data.X_test, data.y_test, sensitive_features=data.X_test[['sensitive_B']])
    assert 'demographic_parity_difference' in metrics_fair
    
    # Test decision tree
    model_dt = Model(algorithm='decision_tree')
    model_dt.train(data.X_train, data.y_train)
    metrics_dt = model_dt.evaluate(data.X_test, data.y_test)
    assert 'accuracy' in metrics_dt
    
    # Test random forest
    model_rf = Model(algorithm='random_forest')
    model_rf.train(data.X_train, data.y_train)
    metrics_rf = model_rf.evaluate(data.X_test, data.y_test)
    assert 'accuracy' in metrics_rf

# Run the test
test_model_training_and_evaluation()