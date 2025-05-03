import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, classification_report, mean_squared_error, r2_score, 
                             mean_absolute_error, matthews_corrcoef, log_loss, balanced_accuracy_score, 
                             roc_curve, precision_recall_curve, brier_score_loss)
from sklearn.calibration import calibration_curve
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate
from scipy.stats import ks_2samp, ttest_ind, permutation_test, chi2_contingency
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime.lime_tabular
from statsmodels.stats.contingency_tables import mcnemar
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class EquiMLEvaluation:
    """Comprehensive evaluation module for the EquiML framework, covering performance, fairness, robustness, and interpretability."""
    
    def evaluate(self, model, X, y_true, y_pred=None, y_pred_proba=None, sensitive_features=None, cv=5, task='classification'):
        """
        Evaluate model performance, fairness, robustness, and interpretability metrics.
        
        Parameters:
        - model: Trained machine learning model
        - X: DataFrame or array of features
        - y_true: Array of true labels or values
        - y_pred: Array of predicted labels/values (optional if model provided)
        - y_pred_proba: Array of predicted probabilities (optional, for classification)
        - sensitive_features: DataFrame of sensitive attribute values (optional)
        - cv: Number of cross-validation folds
        - task: Type of task ('classification' or 'regression')
        
        Returns:
        - metrics: Dictionary containing all computed metrics
        """
        if y_pred is None and model is not None:
            y_pred = model.predict(X)
        if y_pred_proba is None and hasattr(model, 'predict_proba') and task == 'classification':
            y_pred_proba = model.predict_proba(X)

        metrics = {}
        
        # Standard performance metrics
        metrics.update(self.compute_performance_metrics(y_true, y_pred, y_pred_proba, task))
        
        # Calibration metrics (classification only)
        if task == 'classification' and y_pred_proba is not None:
            metrics['calibration'] = self.compute_calibration_metrics(y_true, y_pred_proba)
        
        # Fairness metrics
        if sensitive_features is not None:
            metrics.update(self.compute_fairness_metrics(y_true, y_pred, sensitive_features, task))
        
        # Robustness metrics
        metrics['robustness'] = self.compute_robustness_metrics(model, X, y_true, cv, task)
        
        # Interpretability metrics
        metrics['interpretability'] = self.compute_interpretability_metrics(model, X, y_true, y_pred)
        
        # Statistical tests
        metrics['statistical_tests'] = self.compute_statistical_tests(y_true, y_pred, task)
        
        return metrics
    
    def compute_performance_metrics(self, y_true, y_pred, y_pred_proba, task):
        """Compute a wide range of performance metrics based on task type."""
        metrics = {}
        if task == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            })
            if y_pred_proba is not None:
                n_classes = y_pred_proba.shape[1] if y_pred_proba.ndim > 1 else 2
                if n_classes > 2:
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr')
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
        elif task == 'regression':
            metrics.update({
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            })
        return metrics
    
    def compute_calibration_metrics(self, y_true, y_pred_proba):
        """Compute calibration metrics for probabilistic predictions."""
        n_classes = len(np.unique(y_true))
        calibration_data = {}
        if n_classes == 2:
            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba, n_bins=10)
            calibration_data['binary'] = {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}
        else:
            for i in range(y_pred_proba.shape[1]):
                prob_true, prob_pred = calibration_curve(label_binarize(y_true, classes=np.unique(y_true))[:, i], y_pred_proba[:, i], n_bins=10)
                calibration_data[f'class_{i}'] = {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}
        return {
            'calibration_curve': calibration_data,
            'brier_score': brier_score_loss(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba) if n_classes == 2 else None
        }
    
    def compute_fairness_metrics(self, y_true, y_pred, sensitive_features, task):
        """Compute fairness metrics across sensitive groups."""
        metrics = {}
        if task == 'classification':
            metrics.update({
                'demographic_parity_difference': demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features),
                'equalized_odds_difference': equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features),
                'disparate_impact': self.compute_disparate_impact(y_true, y_pred, sensitive_features)
            })
            mf = MetricFrame(metrics={'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score},
                             y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
            metrics['group_metrics'] = mf.by_group.to_dict()
        elif task == 'regression':
            mf = MetricFrame(metrics={'mse': mean_squared_error, 'mae': mean_absolute_error, 'r2': r2_score},
                             y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
            metrics['group_metrics'] = mf.by_group.to_dict()
            metrics['group_rmse_diff'] = max(mf.by_group['mse']) - min(mf.by_group['mse'])
        return metrics
    
    def compute_disparate_impact(self, y_true, y_pred, sensitive_features):
        """Calculate Disparate Impact ratio for each sensitive feature (classification only)."""
        di_ratios = {}
        for feature in sensitive_features.columns:
            sensitive = sensitive_features[feature]
            groups = sensitive.unique()
            rates = {}
            for group in groups:
                mask = (sensitive == group)
                rates[group] = selection_rate(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 0
            if len(rates) >= 2:
                min_rate = min(rates.values())
                max_rate = max(rates.values())
                di_ratios[feature] = min_rate / max_rate if max_rate > 0 else 0
        return di_ratios
    
    def compute_robustness_metrics(self, model, X, y_true, cv, task):
        """Compute robustness metrics including cross-validation and noise sensitivity."""
        scoring = 'accuracy' if task == 'classification' else 'r2'
        cv_scores = cross_val_score(model, X, y_true, cv=cv, scoring=scoring)
        return {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'noise_sensitivity': self.compute_noise_sensitivity(model, X, y_true, task)
        }
    
    def compute_noise_sensitivity(self, model, X, y_true, task, noise_level=0.1):
        """Evaluate model sensitivity to input noise."""
        X_noisy = X + np.random.normal(0, noise_level, X.shape)
        y_pred_noisy = model.predict(X_noisy)
        return accuracy_score(y_true, y_pred_noisy) if task == 'classification' else r2_score(y_true, y_pred_noisy)
    
    def compute_interpretability_metrics(self, model, X, y_true, y_pred):
        """Compute interpretability metrics like feature importance and SHAP values."""
        metrics = {
            'permutation_importance': self.compute_permutation_importance(model, X, y_true),
            'shap_values': self.compute_shap_values(model, X)
        }
        if isinstance(X, pd.DataFrame):
            metrics['lime_explanations'] = self.compute_lime_explanations(model, X, y_true)
        return metrics
    
    def compute_permutation_importance(self, model, X, y_true):
        """Compute feature importance using permutation importance."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        result = permutation_importance(model, X, y_true, n_repeats=10, random_state=42)
        return {f'feature_{i}': imp for i, imp in enumerate(result.importances_mean)}
    
    def compute_shap_values(self, model, X):
        """Compute SHAP values for interpretability."""
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        return shap_values.values.mean(axis=0).tolist()
    
    def compute_lime_explanations(self, model, X, y_true):
        """Compute LIME explanations for a subset of instances."""
        explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns.tolist())
        exp = explainer.explain_instance(X.iloc[0].values, model.predict_proba, num_features=5)
        return exp.as_list()
    
    def compute_statistical_tests(self, y_true, y_pred, task):
        """Perform statistical tests to compare predictions and ground truth."""
        tests = {
            'ks_test': ks_2samp(y_true, y_pred).statistic,
            't_test': ttest_ind(y_true, y_pred).pvalue,
            'permutation_test': permutation_test((y_true, y_pred), lambda x, y: np.mean(x) - np.mean(y), n_resamples=1000).pvalue
        }
        if task == 'classification' and len(np.unique(y_true)) == 2 and len(np.unique(y_pred)) == 2:
            tests['mcnemar_test'] = self.compute_mcnemar_test(y_true, y_pred)
            tests['chi2_test'] = chi2_contingency(confusion_matrix(y_true, y_pred)).pvalue
        return tests
    
    def compute_mcnemar_test(self, y_true, y_pred):
        """Perform McNemar's test for paired nominal data."""
        table = confusion_matrix(y_true, y_pred)
        result = mcnemar(table, exact=True)
        return result.pvalue
    
    def plot_confusion_matrix(self, y_true, y_pred, filename='confusion_matrix.png'):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(filename)
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, filename='roc_curve.png'):
        """Plot and save ROC curve for binary or multi-class classification."""
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba):.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(filename)
            plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, filename='pr_curve.png'):
        """Plot and save Precision-Recall curve for binary classification."""
        if len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, marker='.', label='Model')
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.savefig(filename)
            plt.close()
    
    def plot_calibration_curve(self, y_true, y_pred_proba, filename='calibration_curve.png'):
        """Plot and save calibration curve for binary or multi-class classification."""
        n_classes = len(np.unique(y_true))
        plt.figure(figsize=(8, 6))
        if n_classes == 2:
            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='Model')
        else:
            for i in range(y_pred_proba.shape[1]):
                prob_true, prob_pred = calibration_curve(label_binarize(y_true, classes=np.unique(y_true))[:, i], y_pred_proba[:, i], n_bins=10)
                plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plot_fairness_metrics(self, metrics, sensitive_feature, filename='fairness_metrics.png'):
        """Plot and save fairness metrics across groups."""
        group_metrics = metrics['group_metrics']
        if 'accuracy' in group_metrics:
            group_accuracy = group_metrics['accuracy'][sensitive_feature]
            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(group_accuracy.keys()), y=list(group_accuracy.values()))
            plt.title(f'Accuracy by {sensitive_feature}')
            plt.savefig(filename)
            plt.close()
    
    def plot_feature_importance(self, importance_dict, filename='feature_importance.png'):
        """Plot and save feature importance."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(importance_dict.values()), y=list(importance_dict.keys()))
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.savefig(filename)
        plt.close()
    
    def plot_shap_summary(self, shap_values, X, filename='shap_summary.png'):
        """Plot and save SHAP summary plot."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(filename)
        plt.close()

    def plot_interactive_roc_curve(self, y_true, y_pred_proba, filename='roc_curve.html'):
        """Generate an interactive ROC curve using Plotly."""
        n_classes = len(np.unique(y_true))
        fig = go.Figure()
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba):.2f})'))
        else:
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            for i in range(y_pred_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Class {i}'))
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        fig.write_html(filename)

    def generate_evaluation_report(self, metrics, filename='evaluation_report.txt'):
        """Generate a comprehensive text report of all evaluation metrics."""
        with open(filename, 'w') as f:
            f.write(f"EquiML Evaluation Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n")
            for category, values in metrics.items():
                f.write(f"\n{category.upper()}:\n")
                if isinstance(values, dict):
                    for key, val in values.items():
                        f.write(f"  {key}: {val}\n")
                else:
                    f.write(f"  {category}: {values}\n")

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    
    # Load data
    X, y = load_iris(return_X_y=True)
    sensitive = pd.DataFrame({'group': np.random.randint(0, 2, size=len(y))})
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Evaluate
    evaluator = EquiMLEvaluation()
    results = evaluator.evaluate(model, X, y, sensitive_features=sensitive, task='classification')
    
    # Generate visualizations and report
    evaluator.plot_confusion_matrix(y, model.predict(X))
    evaluator.plot_roc_curve(y, model.predict_proba(X))
    evaluator.plot_interactive_roc_curve(y, model.predict_proba(X))
    evaluator.generate_evaluation_report(results)
