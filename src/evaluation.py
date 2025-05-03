from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Evaluation:
    """
    A class to evaluate ML models on performance and fairness metrics with visualization.
    
    Attributes:
        model: The trained ML model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        sensitive_features (pd.DataFrame): Sensitive features for fairness evaluation.
        y_pred: Predictions made by the model on X_test.
    """
    
    def __init__(self, model, X_test, y_test, sensitive_features=None):
        """
        Initializes the Evaluation object.
        
        Args:
            model: The trained ML model with a predict method.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            sensitive_features (pd.DataFrame, optional): Sensitive features for fairness evaluation.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features = sensitive_features
        self.y_pred = model.predict(X_test)

    def compute_performance_metrics(self, metrics_list=['accuracy', 'f1_score']):
        """
        Computes specified performance metrics.
        
        Args:
            metrics_list (list): List of metrics to compute (e.g., ['accuracy', 'f1_score']).
        
        Returns:
            dict: Computed performance metrics.
        """
        metrics = {}
        if 'accuracy' in metrics_list:
            metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        if 'f1_score' in metrics_list:
            metrics['f1_score'] = f1_score(self.y_test, self.y_pred, average='binary')
        return metrics

    def compute_fairness_metrics(self, metrics_list=['demographic_parity_difference', 'equalized_odds_difference']):
        """
        Computes specified fairness metrics using MetricFrame.
        
        Args:
            metrics_list (list): List of fairness metrics to compute.
        
        Returns:
            dict: Computed fairness metrics.
        """
        if self.sensitive_features is None:
            raise ValueError("Sensitive features required for fairness metrics.")
        metrics = {}
        if 'demographic_parity_difference' in metrics_list:
            metrics['demographic_parity_difference'] = demographic_parity_difference(
                self.y_test, self.y_pred, sensitive_features=self.sensitive_features
            )
        if 'equalized_odds_difference' in metrics_list:
            metrics['equalized_odds_difference'] = equalized_odds_difference(
                self.y_test, self.y_pred, sensitive_features=self.sensitive_features
            )
        return metrics

    def plot_fairness_metrics(self, metric_name, kind='bar', save_path=None):
        """
        Plots fairness metrics across groups.
        
        Args:
            metric_name (str): Metric to plot (e.g., 'selection_rate').
            kind (str): Type of plot (e.g., 'bar', 'pie').
            save_path (str, optional): Path to save the plot.
        """
        if self.sensitive_features is None:
            raise ValueError("Sensitive features required for plotting.")
        metrics = {'selection_rate': lambda y_true, y_pred: (y_pred == 1).mean()}
        mf = MetricFrame(metrics=metrics, y_true=self.y_test, y_pred=self.y_pred, sensitive_features=self.sensitive_features)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=mf.by_group[metric_name].index, y=mf.by_group[metric_name])
        plt.title(f'{metric_name} by Group')
        plt.xlabel('Group')
        plt.ylabel(metric_name)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def evaluate(self):
        """
        Computes all available performance and fairness metrics.
        
        Returns:
            dict: A report containing all computed metrics.
        """
        performance = self.compute_performance_metrics()
        fairness = self.compute_fairness_metrics() if self.sensitive_features is not None else {}
        return {**performance, **fairness}