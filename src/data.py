import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Data:
    """
    A class to handle data loading, bias analysis, preprocessing, bias mitigation, and splitting for equitable ML.
    
    Attributes:
        dataset_path (str): Path to the dataset file.
        sensitive_features (list): List of column names that are sensitive (e.g., ['gender', 'race']).
        df (pd.DataFrame): The loaded dataset.
        X (pd.DataFrame): Preprocessed features for ML.
        y (pd.Series): Target variable.
        sample_weights (pd.Series): Weights for bias mitigation.
        X_train, X_test, y_train, y_test, sample_weights_train, sample_weights_test: Training and testing splits.
    """
    
    def __init__(self, dataset_path=None, sensitive_features=None):
        """
        Initializes the Data object.
        
        Args:
            dataset_path (str, optional): Path to the dataset file.
            sensitive_features (list, optional): List of sensitive feature names.
        """
        self.dataset_path = dataset_path
        self.sensitive_features = sensitive_features or []
        self.df = None
        self.X = None
        self.y = None
        self.sample_weights = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sample_weights_train = None
        self.sample_weights_test = None

    def load_data(self, dataset_path=None):
        """
        Loads the dataset from the specified path.
        
        Args:
            dataset_path (str, optional): Path to the dataset file. If not provided, uses the initialized path.
        """
        if dataset_path:
            self.dataset_path = dataset_path
        if self.dataset_path:
            self.df = pd.read_csv(self.dataset_path)
        else:
            raise ValueError("No dataset path provided.")

    def analyze_bias(self):
        """
        Analyzes the dataset for potential biases with respect to sensitive features using demographic parity difference.
        
        Returns:
            dict: A report with demographic parity differences for each sensitive feature.
        """
        if not self.df or not self.sensitive_features:
            raise ValueError("Data or sensitive features not set.")
        if self.y is None:
            raise ValueError("Target variable not set. Call preprocess() first.")
        bias_report = {}
        for feature in self.sensitive_features:
            selection_rates = self.df.groupby(feature)[self.y.name].mean()
            dp_diff = selection_rates.max() - selection_rates.min()
            bias_report[feature] = dp_diff
        return bias_report

    def preprocess(self, target_column, categorical_features=None):
        """
        Preprocesses the data by separating features and target, encoding categorical variables, and scaling numerical features.
        
        Args:
            target_column (str): The name of the target variable column.
            categorical_features (list, optional): List of categorical feature names. If None, auto-detects based on dtype.
        """
        if not self.df:
            raise ValueError("No data loaded. Call load_data() first.")
        self.y = self.df[target_column]
        self.X = self.df.drop(columns=[target_column])
        
        # Auto-detect categorical features if not provided
        if categorical_features is None:
            categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
        
        # One-hot encode categorical features
        self.X = pd.get_dummies(self.X, columns=categorical_features, drop_first=True)
        
        # Identify numerical features
        numerical_features = self.X.select_dtypes(include=['number']).columns.tolist()
        
        # Scale numerical features
        scaler = StandardScaler()
        self.X[numerical_features] = scaler.fit_transform(self.X[numerical_features])

    def mitigate_bias(self, method='reweighting'):
        """
        Applies bias mitigation techniques to the dataset, such as reweighting based on sensitive features.
        
        Args:
            method (str): The bias mitigation method to use (currently supports 'reweighting').
        """
        if not self.df or not self.sensitive_features:
            raise ValueError("Data or sensitive features not set.")
        if method == 'reweighting':
            # Compute group sizes based on sensitive features
            group_counts = self.df.groupby(self.sensitive_features).size()
            K = len(group_counts)  # Number of unique groups
            N = len(self.df)
            # Compute group sizes for each sample
            group_sizes = self.df.groupby(self.sensitive_features).transform('size')
            # Compute sample weights: w_i = N / (K * group_size)
            self.sample_weights = N / (K * group_sizes)
        else:
            raise ValueError(f"Unknown mitigation method: {method}")

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets, including sample weights if available.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.
        """
        if not self.X or not self.y:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        if self.sample_weights is not None:
            # Split with sample weights
            self.X_train, self.X_test, self.y_train, self.y_test, self.sample_weights_train, self.sample_weights_test = train_test_split(
                self.X, self.y, self.sample_weights, test_size=test_size, random_state=random_state
            )
        else:
            # Split without sample weights
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
