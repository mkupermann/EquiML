import pandas as pd
from src.data import Data
from src.model import Model
from src.evaluation import EquiMLEvaluation
import os
import warnings

# Suppress the specific UserWarning from sklearn
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LogisticRegression was fitted with feature names")


def test_framework():
    # Step 1: Initialize and load data
    data = Data(dataset_path='tests/adult.csv', sensitive_features=['sex'])
    data.load_data()
    data.preprocess(
        target_column='income',
        numerical_features=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
        categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    )
    data.split_data(test_size=0.2, random_state=42)

    # The sensitive feature 'sex' is one-hot encoded. Let's find the column name.
    # It will be something like 'sex_Male'.
    sensitive_feature_column = [col for col in data.X_train.columns if col.startswith('sex_')]
    if not sensitive_feature_column:
        raise ValueError("Sensitive feature column not found in preprocessed data.")
    sensitive_feature_column = sensitive_feature_column[0]

    sensitive_features_train = data.X_train[sensitive_feature_column]
    X_train = data.X_train.drop(columns=[sensitive_feature_column])

    sensitive_features_test = data.X_test[sensitive_feature_column]
    X_test = data.X_test.drop(columns=[sensitive_feature_column])


    # Step 2: Train a model with fairness constraints
    model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')
    model.train(X_train, data.y_train, sensitive_features=sensitive_features_train)

    # Step 3: Evaluate the model
    evaluation = EquiMLEvaluation()
    # Use cv=2 to avoid errors with small datasets
    metrics = evaluation.evaluate(model, X_test, data.y_test, sensitive_features=sensitive_features_test, cv=2)

    # Step 4: Generate report
    report_path = 'test_report.html'
    template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')
    evaluation.generate_report(metrics, output_path=report_path, template_path=template_path)
    assert os.path.exists(report_path)
    os.remove(report_path)

    # Step 5: Assertions
    assert 'accuracy' in metrics
    assert 'demographic_parity_difference' in metrics
    assert 'robustness' in metrics
    assert 'interpretability' in metrics

# Run the test
test_framework()