from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import demographic_parity_difference

class Model:
    """
    A class to handle the training, prediction, and evaluation of machine learning models with fairness constraints.
    
    Attributes:
        algorithm (str): The machine learning algorithm to use (e.g., 'logistic_regression', 'decision_tree', 'random_forest').
        fairness_constraint (str): The fairness constraint to apply (e.g., 'demographic_parity').
        model: The underlying machine learning model.
    """
    
    def __init__(self, algorithm='logistic_regression', fairness_constraint=None):
        """
        Initializes the Model object.
        
        Args:
            algorithm (str): The machine learning algorithm to use.
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
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        if self.fairness_constraint == 'demographic_parity':
            if sensitive_features is None:
                raise ValueError("Sensitive features are required for fairness constraints.")
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

    def evaluate(self, X_test, y_test, sensitive_features=None):
        """
        Evaluates the model's performance and fairness.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            sensitive_features (pd.DataFrame): Sensitive features for fairness evaluation.
        
        Returns:
            dict: A report containing performance and fairness metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        y_pred = self.predict(X_test)
        performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='binary')
        }
        if sensitive_features is not None:
            fairness_metrics = {
                'demographic_parity_difference': demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
            }
        else:
            fairness_metrics = {}
        return {**performance_metrics, **fairness_metrics}