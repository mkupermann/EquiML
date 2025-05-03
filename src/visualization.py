import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, calibration_curve
import numpy as np
import pandas as pd
from shap import summary_plot

def plot_group_metrics(metrics, metric_names, sensitive_feature, title=None, save_path=None, kind='bar', palette='viridis'):
    """
    Plot bar or line chart for multiple group-wise metrics.
    
    Parameters:
    - metrics: Dictionary containing group-wise metrics.
    - metric_names: List of metric names to plot (e.g., ['accuracy', 'precision']).
    - sensitive_feature: Name of the sensitive feature (e.g., 'gender').
    - title: Optional title for the plot.
    - save_path: Optional path to save the plot.
    - kind: Type of plot ('bar' or 'line').
    - palette: Color palette for the plot.
    """
    if 'group_metrics' not in metrics or sensitive_feature not in metrics['group_metrics']:
        raise ValueError(f"No group metrics found for sensitive feature '{sensitive_feature}'.")
    
    group_metrics = metrics['group_metrics'][sensitive_feature]
    groups = list(group_metrics.keys())
    
    plt.figure(figsize=(12, 8))
    for metric in metric_names:
        if metric not in group_metrics:
            raise ValueError(f"Metric '{metric}' not found in group metrics for '{sensitive_feature}'.")
        values = list(group_metrics[metric].values())
        if kind == 'bar':
            sns.barplot(x=groups, y=values, palette=palette, label=metric)
        elif kind == 'line':
            sns.lineplot(x=groups, y=values, marker='o', label=metric)
        else:
            raise ValueError(f"Unsupported plot kind: {kind}")
    
    plt.title(title or f'Metrics by {sensitive_feature.capitalize()}')
    plt.xlabel(sensitive_feature.capitalize())
    plt.ylabel('Metric Value')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    """
    Plot heatmap for confusion matrix.
    
    Parameters:
    - y_true: True labels (array-like or Series).
    - y_pred: Predicted labels (array-like or Series).
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve_interactive(y_true, y_pred_proba, title='ROC Curve', save_path=None):
    """
    Plot interactive ROC curve using Plotly.
    
    Parameters:
    - y_true: True labels (array-like or Series).
    - y_pred_proba: Predicted probabilities (array-like or Series).
    - title: Title for the plot.
    - save_path: Optional path to save the plot as HTML.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
        raise ValueError("Interactive ROC curve currently supports only binary classification.")
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    if save_path:
        fig.write_html(save_path)
    fig.show()

def plot_precision_recall_curve(y_true, y_pred_proba, title='Precision-Recall Curve', save_path=None):
    """
    Plot precision-recall curve for binary classification.
    
    Parameters:
    - y_true: True labels (array-like or Series).
    - y_pred_proba: Predicted probabilities (array-like or Series).
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Precision-Recall curve is only for binary classification.")
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Model')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, title='Calibration Curve', save_path=None):
    """
    Plot calibration curve for probability predictions.
    
    Parameters:
    - y_true: True labels (array-like or Series).
    - y_pred_proba: Predicted probabilities (array-like or Series).
    - n_bins: Number of bins for calibration.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
        raise ValueError("Calibration curve is only for binary classification.")
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.title(title)
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_shap_summary(shap_values, X, feature_names=None, title='SHAP Summary Plot', save_path=None):
    """
    Plot SHAP summary plot for feature importance.
    
    Parameters:
    - shap_values: SHAP values from explainer.
    - X: DataFrame or array of features.
    - feature_names: List of feature names (optional).
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(np.array(X).shape[1])]
    summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(importance_dict, title='Feature Importance', save_path=None, orientation='horizontal'):
    """
    Plot bar chart for feature importance.
    
    Parameters:
    - importance_dict: Dictionary of feature names and their importance scores.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    - orientation: 'horizontal' or 'vertical' bar plot.
    """
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    plt.figure(figsize=(10, 6))
    if orientation == 'horizontal':
        sns.barplot(x=importance, y=features, palette='coolwarm')
    else:
        sns.barplot(x=features, y=importance, palette='coolwarm')
        plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel('Importance' if orientation == 'horizontal' else 'Feature')
    plt.ylabel('Feature' if orientation == 'horizontal' else 'Importance')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_interactive_fairness_dashboard(metrics, sensitive_features, save_path=None):
    """
    Create an interactive dashboard for fairness metrics using Plotly.
    
    Parameters:
    - metrics: Dictionary containing fairness metrics.
    - sensitive_features: List of sensitive feature names.
    - save_path: Optional path to save the dashboard as HTML.
    """
    if 'group_metrics' not in metrics:
        raise ValueError("No group metrics found in the provided metrics dictionary.")
    fig = go.Figure()
    for feature in sensitive_features:
        if feature not in metrics['group_metrics']:
            continue
        group_metrics = metrics['group_metrics'][feature]
        df = pd.DataFrame.from_dict(group_metrics, orient='index')
        for metric in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df[metric], name=f'{feature} - {metric}'))
    fig.update_layout(barmode='group', title='Fairness Metrics by Group', xaxis_title='Group', yaxis_title='Metric Value')
    if save_path:
        fig.write_html(save_path)
    fig.show()

def plot_data_distribution(X, feature, sensitive_feature=None, title=None, save_path=None):
    """
    Plot distribution of a feature, optionally across sensitive groups.
    
    Parameters:
    - X: DataFrame of features.
    - feature: Feature to plot.
    - sensitive_feature: Optional sensitive feature to group by.
    - title: Optional title for the plot.
    - save_path: Optional path to save the plot.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a Pandas DataFrame.")
    plt.figure(figsize=(10, 6))
    if sensitive_feature:
        sns.histplot(data=X, x=feature, hue=sensitive_feature, multiple='stack', palette='Set2')
    else:
        sns.histplot(data=X, x=feature, palette='Set2')
    plt.title(title or f'Distribution of {feature}' + (f' by {sensitive_feature}' if sensitive_feature else ''))
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_heatmap(X, title='Feature Correlation Heatmap', save_path=None, method='pearson'):
    """
    Plot heatmap for feature correlations.
    
    Parameters:
    - X: DataFrame of features.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    - method: Correlation method ('pearson', 'spearman', 'kendall').
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a Pandas DataFrame.")
    corr = X.corr(method=method)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning Curve', save_path=None):
    """
    Plot learning curve for model training.
    
    Parameters:
    - train_sizes: Array of training set sizes.
    - train_scores: Array of training scores.
    - test_scores: Array of test scores.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Train', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, test_mean, label='Validation', marker='o')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
