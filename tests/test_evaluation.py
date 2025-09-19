import pandas as pd
import numpy as np
from src.model import Model
from src.data import Data
from src.evaluation import EquiMLEvaluation
import pytest
import os

@pytest.fixture
def dummy_data():
    """Provides a dummy dataset for evaluation testing."""
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'sensitive': np.random.choice(['A', 'B'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    data = Data(sensitive_features=['sensitive'])
    data.df = df
    data.preprocess(target_column='target', numerical_features=['feature1'], categorical_features=['sensitive'])
    data.split_data()
    return data

@pytest.fixture
def trained_model(dummy_data):
    """Provides a trained model."""
    model = Model(algorithm='logistic_regression')
    model.train(dummy_data.X_train, dummy_data.y_train)
    return model

def test_evaluate(trained_model, dummy_data):
    """Tests the evaluate method."""
    evaluation = EquiMLEvaluation()
    metrics = evaluation.evaluate(
        trained_model,
        dummy_data.X_test,
        dummy_data.y_test,
        sensitive_features=dummy_data.X_test['sensitive_B']
    )
    assert 'accuracy' in metrics
    assert 'demographic_parity_difference' in metrics
    assert 'equal_opportunity_difference' in metrics
    assert 'robustness' in metrics
    assert 'interpretability' in metrics

def test_compute_fairness_metrics(trained_model, dummy_data):
    """Tests the compute_fairness_metrics method."""
    evaluation = EquiMLEvaluation()
    y_pred = trained_model.predict(dummy_data.X_test)
    fairness_metrics = evaluation.compute_fairness_metrics(
        dummy_data.y_test,
        y_pred,
        dummy_data.X_test['sensitive_B'],
        task='classification'
    )
    assert 'demographic_parity_difference' in fairness_metrics
    assert 'equalized_odds_difference' in fairness_metrics
    assert 'equal_opportunity_difference' in fairness_metrics

def test_plot_model_comparison(trained_model, dummy_data):
    """Tests the plot_model_comparison method."""
    evaluation = EquiMLEvaluation()
    metrics1 = evaluation.evaluate(trained_model, dummy_data.X_test, dummy_data.y_test)
    metrics2 = evaluation.evaluate(trained_model, dummy_data.X_test, dummy_data.y_test)
    results_dict = {'model1': metrics1, 'model2': metrics2}

    output_filename = "test_comparison.png"
    evaluation.plot_model_comparison(results_dict, output_filename=output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_generate_report(trained_model, dummy_data):
    """Tests the generate_report method."""
    evaluation = EquiMLEvaluation()
    metrics = evaluation.evaluate(trained_model, dummy_data.X_test, dummy_data.y_test)

    # Create dummy plot files
    for plot_file in ['confusion_matrix.png', 'roc_curve.png', 'fairness_metrics.png', 'feature_importance.png', 'shap_summary.png']:
        with open(plot_file, 'w') as f:
            f.write("dummy plot")

    report_path = 'test_report.html'
    evaluation.generate_report(metrics, output_path=report_path, template_path='src/report_template.html')

    assert os.path.exists(report_path)
    with open(report_path, 'r') as f:
        content = f.read()
        assert "EquiML Evaluation Report" in content
        assert "Accuracy" in content

    # Clean up
    os.remove(report_path)
    for plot_file in ['confusion_matrix.png', 'roc_curve.png', 'fairness_metrics.png', 'feature_importance.png', 'shap_summary.png']:
        os.remove(plot_file)
