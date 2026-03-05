"""Test the core audit pipeline end-to-end."""

import pytest
import pandas as pd
import numpy as np
import os


@pytest.fixture
def adult_sample():
    """Minimal synthetic dataset mimicking the Adult Census structure."""
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


class TestDataModule:
    def test_preprocess(self, adult_sample):
        from equiml.data import Data

        data = Data(sensitive_features=["gender"])
        data.df = adult_sample
        data.preprocess(
            target_column="income",
            numerical_features=["age", "hours_per_week"],
            categorical_features=["gender"],
        )
        assert data.X is not None
        assert data.y is not None
        assert len(data.X) == len(data.y)

    def test_split(self, adult_sample):
        from equiml.data import Data

        data = Data(sensitive_features=["gender"])
        data.df = adult_sample
        data.preprocess(target_column="income", numerical_features=["age", "hours_per_week"])
        data.split_data(test_size=0.2)
        assert len(data.X_train) > len(data.X_test)

    def test_reweighing(self, adult_sample):
        from equiml.data import Data

        data = Data(sensitive_features=["gender"])
        data.df = adult_sample
        data.preprocess(
            target_column="income",
            numerical_features=["age", "hours_per_week"],
            categorical_features=["gender"],
        )
        data.mitigate_bias(method="reweighing")
        assert data.sample_weights is not None
        assert len(data.sample_weights) == len(data.y)


class TestModelModule:
    def test_train_predict(self, adult_sample):
        from equiml.data import Data
        from equiml.model import Model

        data = Data(sensitive_features=["gender"])
        data.df = adult_sample
        data.preprocess(target_column="income", numerical_features=["age", "hours_per_week"],
                       categorical_features=["gender"])
        data.split_data()

        model = Model(algorithm="logistic_regression")
        model.train(data.X_train, data.y_train)
        preds = model.predict(data.X_test)
        assert len(preds) == len(data.X_test)

    def test_fair_training(self, adult_sample):
        from equiml.data import Data
        from equiml.model import Model

        data = Data(sensitive_features=["gender"])
        data.df = adult_sample
        data.preprocess(
            target_column="income",
            numerical_features=["age", "hours_per_week"],
            categorical_features=["gender"],
        )
        data.split_data()

        sensitive_cols = [c for c in data.X_train.columns if "gender" in c]
        sensitive_train = data.X_train[sensitive_cols[0]] if sensitive_cols else None

        model = Model(algorithm="logistic_regression", fairness_constraint="demographic_parity")
        model.train(data.X_train, data.y_train, sensitive_features=sensitive_train)
        preds = model.predict(data.X_test)
        assert len(preds) == len(data.X_test)


class TestEvaluation:
    def test_evaluate(self, adult_sample):
        from equiml.data import Data
        from equiml.model import Model
        from equiml.evaluation import EquiMLEvaluation

        data = Data(sensitive_features=["gender"])
        data.df = adult_sample
        data.preprocess(
            target_column="income",
            numerical_features=["age", "hours_per_week"],
            categorical_features=["gender"],
        )
        data.split_data()

        model = Model(algorithm="random_forest")
        model.train(data.X_train, data.y_train)
        preds = model.predict(data.X_test)

        sensitive_cols = [c for c in data.X_test.columns if "gender" in c]
        sensitive_test = data.X_test[sensitive_cols[0]] if sensitive_cols else None

        evaluator = EquiMLEvaluation()
        metrics = evaluator.evaluate(
            model, data.X_test, data.y_test,
            y_pred=preds,
            sensitive_features=sensitive_test,
        )

        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "demographic_parity_difference" in metrics


class TestMonitoring:
    def test_bias_monitor(self):
        from equiml.monitoring import BiasMonitor

        monitor = BiasMonitor(sensitive_features=["gender"])
        predictions = np.array([1, 0, 1, 1, 0, 0, 1, 1])
        sensitive = pd.DataFrame({"gender": ["M", "F", "M", "F", "M", "F", "M", "F"]})
        result = monitor.monitor_predictions(predictions, sensitive)

        assert "metrics" in result
        assert "violations" in result

    def test_drift_detector(self):
        from equiml.monitoring import DriftDetector

        np.random.seed(42)
        ref = pd.DataFrame({"feat": np.random.randn(100)})
        new = pd.DataFrame({"feat": np.random.randn(100) + 2})  # drifted

        detector = DriftDetector(reference_data=ref)
        result = detector.detect_drift(new)

        assert result["drift_detected"] is True
