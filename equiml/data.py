import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Data:
    """
    Handles tabular data loading, preprocessing, outlier detection, splitting,
    and bias-mitigation preprocessing for the EquiML audit pipeline.

    Tabular classification only — no text, image, or time-series support.

    Attributes:
        dataset_path (str): Path to the dataset file.
        sensitive_features (list): List of column names that are sensitive (e.g., ['gender', 'race']).
        df (pd.DataFrame): The loaded dataset.
        X (pd.DataFrame): Preprocessed features for ML.
        y (pd.Series): Target variable.
        X_train, X_test, y_train, y_test: Training and testing splits.
    """

    def __init__(self, dataset_path: Optional[str] = None, sensitive_features: Optional[List[str]] = None):
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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sensitive_df = None
        self.sample_weights = None
        self.sample_weights_train = None
        self.sample_weights_test = None

    def load_data(self, dataset_path: Optional[str] = None) -> None:
        """
        Loads the dataset from the specified path. Supports CSV, JSON, Excel, and Parquet files.

        Args:
            dataset_path (str, optional): Path to the dataset file. If not provided, uses the initialized path.

        Raises:
            ValueError: If no dataset path is provided or the file format is unsupported.
        """
        if dataset_path:
            self.dataset_path = dataset_path
        if not self.dataset_path:
            raise ValueError("No dataset path provided.")

        try:
            if self.dataset_path.endswith('.csv'):
                self.df = pd.read_csv(self.dataset_path)
            elif self.dataset_path.endswith('.json'):
                self.df = pd.read_json(self.dataset_path)
            elif self.dataset_path.endswith('.xlsx'):
                self.df = pd.read_excel(self.dataset_path)
            elif self.dataset_path.endswith('.parquet'):
                self.df = pd.read_parquet(self.dataset_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV, JSON, Excel, or Parquet files.")
            logger.info(f"Data loaded successfully from {self.dataset_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def preprocess(self, target_column: str, numerical_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, impute_strategy: str = 'mean', scaling: str = 'standard') -> None:
        """
        Preprocesses tabular data: handles missing values, encodes categorical
        variables, and scales numerical features.

        Args:
            target_column (str): The name of the target variable column.
            numerical_features (list, optional): List of numerical feature names.
            categorical_features (list, optional): List of categorical feature names.
            impute_strategy (str): Imputation strategy for numerical features ('mean', 'median', 'constant').
            scaling (str): Scaling method for numerical features ('standard', 'minmax', 'robust').

        Raises:
            ValueError: If data is not loaded or required columns are missing.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        le = LabelEncoder()
        self.y = pd.Series(le.fit_transform(self.df[target_column]), name=target_column)
        self.X = self.df.drop(columns=[target_column])

        if self.sensitive_features:
            self.sensitive_df = self.X[self.sensitive_features].copy()

        # Handle missing values
        if numerical_features:
            num_imputer = SimpleImputer(strategy=impute_strategy)
            self.X[numerical_features] = num_imputer.fit_transform(self.X[numerical_features])
        if categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.X[categorical_features] = cat_imputer.fit_transform(self.X[categorical_features])

        # Encode categorical features
        if categorical_features:
            encoder = OneHotEncoder(drop='first', sparse_output=False)
            encoded_cats = encoder.fit_transform(self.X[categorical_features])
            cat_columns = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_cats, columns=cat_columns, index=self.X.index)
            self.X = pd.concat([self.X.drop(columns=categorical_features), encoded_df], axis=1)

        # Scale numerical features
        if numerical_features:
            if scaling == 'standard':
                scaler = StandardScaler()
            elif scaling == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif scaling == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {scaling}")
            self.X[numerical_features] = scaler.fit_transform(self.X[numerical_features])

    def detect_outliers(self, features: Optional[List[str]] = None, method: str = 'zscore', threshold: float = 3.0, action: str = 'flag') -> Optional[pd.DataFrame]:
        """
        Detects and optionally handles outliers in the specified features.

        Args:
            features (list, optional): List of features to check for outliers. Defaults to all numerical features.
            method (str): Method to use ('zscore' or 'iqr').
            threshold (float): Threshold for detecting outliers.
            action (str): Action to take ('flag', 'remove', 'cap').

        Returns:
            pd.DataFrame: DataFrame indicating outliers (if action='flag').

        Raises:
            ValueError: If features or method are invalid.
        """
        if self.X is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        if features is None:
            features = self.X.select_dtypes(include=[np.number]).columns
        elif not set(features).issubset(self.X.columns):
            raise ValueError("Some features not found in dataset.")

        outliers = pd.DataFrame(index=self.X.index)

        for feature in features:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(self.X[feature], nan_policy='omit'))
                outliers[feature] = z_scores > threshold
            elif method == 'iqr':
                Q1 = self.X[feature].quantile(0.25)
                Q3 = self.X[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers[feature] = (self.X[feature] < (Q1 - 1.5 * IQR)) | (self.X[feature] > (Q3 + 1.5 * IQR))
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")

        if action == 'remove':
            mask = ~outliers.any(axis=1)
            self.X = self.X[mask]
            self.y = self.y[mask]
            logger.info("Outliers removed from dataset.")
        elif action == 'cap':
            for feature in features:
                if method == 'zscore':
                    self.X[feature] = self.X[feature].clip(-threshold, threshold)
                elif method == 'iqr':
                    Q1 = self.X[feature].quantile(0.25)
                    Q3 = self.X[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    self.X[feature] = self.X[feature].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            logger.info("Outliers capped in dataset.")
        elif action == 'flag':
            return outliers
        else:
            raise ValueError(f"Unsupported action: {action}")

    def mitigate_bias(self, method='reweighing'):
        """
        Applies bias mitigation techniques to the data. Currently supports 'reweighing'.

        Args:
            method (str): The mitigation method to use.
        """
        if method == 'reweighing':
            if self.y is None or self.sensitive_df is None:
                raise ValueError("y or sensitive_features not set. Call preprocess() first.")

            self.sample_weights = pd.Series(1.0, index=self.y.index)
            # Assuming a single sensitive feature for now
            sensitive_col = self.sensitive_df.columns[0]

            # Calculate weights
            for y_val in self.y.unique():
                for s_val in self.sensitive_df[sensitive_col].unique():
                    y_mask = (self.y == y_val)
                    s_mask = (self.sensitive_df[sensitive_col] == s_val)

                    p_y = y_mask.mean()
                    p_s = s_mask.mean()
                    p_ys = (y_mask & s_mask).mean()

                    weight = (p_y * p_s) / p_ys if p_ys > 0 else 1.0

                    self.sample_weights[y_mask & s_mask] = weight

            logger.info("Sample weights computed using reweighing.")
        else:
            raise ValueError(f"Unsupported mitigation method: {method}")

    def apply_bias_mitigation(self, method: str = 'reweighing') -> None:
        """
        Apply bias mitigation preprocessing techniques.

        Args:
            method (str): Bias mitigation method ('reweighing', 'correlation_removal')
        """
        if self.sensitive_features is None or len(self.sensitive_features) == 0:
            logger.warning("No sensitive features specified. Skipping bias mitigation.")
            return

        try:
            if method == 'reweighing':
                self._apply_reweighing()
            elif method == 'correlation_removal':
                self._apply_correlation_removal()
            else:
                raise ValueError(f"Unsupported bias mitigation method: {method}")
            logger.info(f"Applied bias mitigation: {method}")
        except Exception as e:
            logger.error(f"Bias mitigation failed: {str(e)}")
            raise

    def _apply_reweighing(self) -> None:
        """Apply reweighing to balance sensitive groups."""
        if self.X is None or self.y is None:
            raise ValueError("Data must be preprocessed before applying reweighing.")

        # Calculate sample weights to balance sensitive groups
        from sklearn.utils.class_weight import compute_sample_weight

        # Get sensitive feature values (exact-prefix match: a column matches
        # sensitive feature `sf` iff col == sf or col.startswith(f"{sf}_"))
        sensitive_cols = [
            col for col in self.X.columns
            if any(col == sf or col.startswith(f"{sf}_") for sf in self.sensitive_features)
        ]
        if not sensitive_cols:
            logger.warning("No sensitive feature columns found for reweighing.")
            return

        # Use first sensitive feature for reweighing
        sensitive_values = self.X[sensitive_cols[0]]

        # Create stratification based on sensitive feature and target
        stratify_values = [f"{s}_{t}" for s, t in zip(sensitive_values, self.y)]

        # Calculate balanced weights
        self.sample_weights = compute_sample_weight('balanced', stratify_values)
        logger.info("Reweighing applied successfully.")

    def _apply_correlation_removal(self) -> None:
        """Remove correlation with sensitive attributes."""
        try:
            from fairlearn.preprocessing import CorrelationRemover

            # Find sensitive feature columns (exact-prefix match)
            sensitive_cols = [
                col for col in self.X.columns
                if any(col == sf or col.startswith(f"{sf}_") for sf in self.sensitive_features)
            ]
            if not sensitive_cols:
                logger.warning("No sensitive feature columns found for correlation removal.")
                return

            # Get sensitive feature indices
            sensitive_indices = [self.X.columns.get_loc(col) for col in sensitive_cols]

            # Apply correlation removal
            cr = CorrelationRemover(sensitive_feature_ids=sensitive_indices)
            X_transformed = cr.fit_transform(self.X)

            # Update features with transformed data
            self.X = pd.DataFrame(X_transformed, columns=self.X.columns, index=self.X.index)
            logger.info("Correlation removal applied successfully.")

        except ImportError:
            logger.error("Fairlearn not available. Install with: pip install fairlearn")
            raise

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.

        Raises:
            ValueError: If data is not preprocessed.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")

        if self.sample_weights is not None:
            self.X_train, self.X_test, self.y_train, self.y_test, self.sample_weights_train, self.sample_weights_test = train_test_split(
                self.X, self.y, self.sample_weights, test_size=test_size, random_state=random_state
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
        logger.info("Data split into training and testing sets.")
