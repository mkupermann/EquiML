from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import shap
import numpy as np

class Model:
    """
    A class to handle training, prediction, evaluation, and explanation of ML models with fairness constraints.
    
    Attributes:
        algorithm (str): ML algorithm (e.g., 'logistic_regression', 'decision_tree', 'random_forest', 'svm').
        fairness_constraint (str): Fairness constraint (e.g., 'demographic_parity').
        model: The underlying ML model.
    """
    
    def __init__(self, algorithm='logistic_regression', fairness_constraint=None):
        """
        Initializes the Model object.
        
        Args:
            algorithm (str): The ML algorithm to use.
            fairness_constraint (str): The fairness constraint to apply during training.
        """
        self.algorithm = algorithm
        self.fairness_constraint = fairness_constraint
        self.model = None

    def train(self, X_train, y_train, sensitive_features=None, sample_weights=None):
        """
        Trains the model with optional fairness constraints and sample weights.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            sensitive_features (pd.DataFrame): Sensitive features for fairness constraints.
            sample_weights (pd.Series, optional): Sample weights for bias mitigation.
        """
        if self.algorithm == 'logistic_regression':
            base_model = LogisticRegression()
        elif self.algorithm == 'decision_tree':
            base_model = DecisionTreeClassifier()
        elif self.algorithm == 'random_forest':
            base_model = RandomForestClassifier()
        elif self.algorithm == 'svm':
            base_model = SVC(probability=True)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        if self.fairness_constraint == 'demographic_parity':
            if sensitive_features is None:
                raise ValueError("Sensitive features required for fairness constraints.")
            constraint = DemographicParity()
            self.model = ExponentiatedGradient(base_model, constraint)
            self.model.fit(X_train, y_train, sensitive_features=sensitive_features)
        else:
            self.model = base_model
            self.model.fit(X_train, y_train, sample_weight=sample_weights)

    def predict(self, X):
        """
        Makes predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Features to predict on.
        
        Returns:
            np.array: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def explain_prediction(self, X_instance, background_data=None):
        """
        Explains a prediction using SHAP values.
        
        Args:
            X_instance (pd.DataFrame): Single instance to explain.
            background_data (pd.DataFrame, optional): Background data for SHAP explainer.
        
        Returns:
            shap.Explanation: SHAP values for the prediction.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if background_data is None:
            background_data = X_instance.sample(min(100, len(X_instance)), random_state=42)
        explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
        shap_values = explainer.shap_values(X_instance)
        return shap_values