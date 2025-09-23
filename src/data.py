import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy import stats
import arff
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import cv2  # For image data support
import logging
from typing import List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _setup_nltk():
    """
    Setup NLTK data downloads if not already present.
    This function should be called before using text processing features.
    """
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading required NLTK data...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            logger.info("NLTK data download completed.")
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}. Text processing may not work properly.")

class Data:
    """
    A class to handle data loading, preprocessing, outlier detection, multi-modal data support,
    text processing, feature engineering, and splitting for equitable ML in the EquiML framework.
    
    Attributes:
        dataset_path (str): Path to the dataset file.
        sensitive_features (list): List of column names that are sensitive (e.g., ['gender', 'race']).
        text_features (list): List of text feature names for processing.
        image_features (list): List of image feature names or paths for processing.
        df (pd.DataFrame): The loaded dataset.
        X (pd.DataFrame): Preprocessed features for ML.
        y (pd.Series): Target variable.
        X_train, X_test, y_train, y_test: Training and testing splits.
    """
    
    def __init__(self, dataset_path: Optional[str] = None, sensitive_features: Optional[List[str]] = None, text_features: Optional[List[str]] = None, image_features: Optional[List[str]] = None):
        """
        Initializes the Data object.
        
        Args:
            dataset_path (str, optional): Path to the dataset file.
            sensitive_features (list, optional): List of sensitive feature names.
            text_features (list, optional): List of text feature names for processing.
            image_features (list, optional): List of image feature names or paths for processing.
        """
        self.dataset_path = dataset_path
        self.sensitive_features = sensitive_features or []
        self.text_features = text_features or []
        self.image_features = image_features or []
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
        Loads the dataset from the specified path. Supports CSV, JSON, Excel, Parquet, and ARFF files.
        
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
            elif self.dataset_path.endswith('.arff'):
                with open(self.dataset_path, 'r') as f:
                    data = arff.load(f)
                self.df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
            else:
                raise ValueError("Unsupported file format. Please use CSV, JSON, Excel, Parquet or ARFF files.")
            logger.info(f"Data loaded successfully from {self.dataset_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def preprocess(self, target_column: str, numerical_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, impute_strategy: str = 'mean', scaling: str = 'standard') -> None:
        """
        Preprocesses the data by handling missing values, encoding categorical variables,
        scaling numerical features, processing text features, and handling image data.
        
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
        
        # Process text features
        if self.text_features:
            for feature in self.text_features:
                if feature not in self.X.columns:
                    raise ValueError(f"Text feature '{feature}' not found in dataset.")
                self.X[feature] = self.X[feature].apply(self._clean_text)
                vectorizer = TfidfVectorizer(max_features=100)
                text_features = vectorizer.fit_transform(self.X[feature]).toarray()
                text_df = pd.DataFrame(text_features, columns=[f"{feature}_{i}" for i in range(100)], index=self.X.index)
                self.X = pd.concat([self.X.drop(columns=[feature]), text_df], axis=1)
        
        # Process image features (assuming paths to images are provided)
        if self.image_features:
            for feature in self.image_features:
                if feature not in self.X.columns:
                    raise ValueError(f"Image feature '{feature}' not found in dataset.")
                image_data = self._process_images(self.X[feature])
                image_df = pd.DataFrame(image_data, columns=[f"{feature}_pixel_{i}" for i in range(image_data.shape[1])], index=self.X.index)
                self.X = pd.concat([self.X.drop(columns=[feature]), image_df], axis=1)
        
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

    def _clean_text(self, text: str) -> str:
        """
        Helper function to clean text data.

        Args:
            text (str): Raw text input.

        Returns:
            str: Cleaned text.
        """
        try:
            _setup_nltk()  # Ensure NLTK data is available
            text = str(text).lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in stopwords.words('english')]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            return ""

    def _process_images(self, image_paths: pd.Series, target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
        """
        Helper function to process image data into flattened arrays.
        
        Args:
            image_paths (pd.Series): Series of image file paths.
            target_size (tuple): Desired image size (width, height).
        
        Returns:
            np.ndarray: Flattened image data.
        """
        processed_images = []
        for path in image_paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Grayscale for simplicity
                if img is not None:
                    img_resized = cv2.resize(img, target_size)
                    processed_images.append(img_resized.flatten())
                else:
                    processed_images.append(np.zeros(target_size[0] * target_size[1]))
            except Exception as e:
                logger.warning(f"Failed to process image '{path}': {str(e)}")
                processed_images.append(np.zeros(target_size[0] * target_size[1]))
        return np.array(processed_images)

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

    def feature_engineering(self, polynomial_degree: int = 2, interaction_only: bool = False, include_log: bool = False) -> None:
        """
        Performs feature engineering by creating polynomial features, interaction terms, and optional log transformations.
        
        Args:
            polynomial_degree (int): Degree of polynomial features.
            interaction_only (bool): If True, only interaction features are created.
            include_log (bool): If True, adds log-transformed features for numerical columns.
        
        Raises:
            ValueError: If dataset is not preprocessed or contains invalid data for transformations.
        """
        if self.X is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        if not numerical_cols.any():
            raise ValueError("No numerical columns found for feature engineering.")
        
        poly = PolynomialFeatures(degree=polynomial_degree, interaction_only=interaction_only)
        poly_features = poly.fit_transform(self.X[numerical_cols])
        poly_columns = poly.get_feature_names_out(numerical_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_columns, index=self.X.index)
        
        if include_log:
            for col in numerical_cols:
                if (self.X[col] <= 0).any():
                    logger.warning(f"Skipping log transformation for '{col}' due to non-positive values.")
                    continue
                poly_df[f'log_{col}'] = np.log1p(self.X[col])
        
        self.X = pd.concat([self.X.drop(columns=numerical_cols), poly_df], axis=1)
        logger.info("Feature engineering completed.")

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
            method (str): Bias mitigation method ('reweighing', 'correlation_removal', 'data_augmentation')
        """
        if self.sensitive_features is None or len(self.sensitive_features) == 0:
            logger.warning("No sensitive features specified. Skipping bias mitigation.")
            return

        try:
            if method == 'reweighing':
                self._apply_reweighing()
            elif method == 'correlation_removal':
                self._apply_correlation_removal()
            elif method == 'data_augmentation':
                self._apply_data_augmentation()
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

        # Get sensitive feature values
        sensitive_cols = [col for col in self.X.columns if any(sf in col for sf in self.sensitive_features)]
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

            # Find sensitive feature columns
            sensitive_cols = [col for col in self.X.columns if any(sf in col for sf in self.sensitive_features)]
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

    def _apply_data_augmentation(self) -> None:
        """Apply data augmentation for underrepresented groups."""
        if self.X is None or self.y is None:
            raise ValueError("Data must be preprocessed before applying data augmentation.")

        # Find sensitive feature columns
        sensitive_cols = [col for col in self.X.columns if any(sf in col for sf in self.sensitive_features)]
        if not sensitive_cols:
            logger.warning("No sensitive feature columns found for data augmentation.")
            return

        # Use SMOTE for minority groups based on sensitive features
        try:
            from imblearn.over_sampling import SMOTE

            # Combine target and sensitive feature for stratification
            sensitive_values = self.X[sensitive_cols[0]]
            combined_target = [f"{s}_{t}" for s, t in zip(sensitive_values, self.y)]

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(self.X, combined_target)

            # Extract original target from combined target
            y_original = [int(ct.split('_')[1]) for ct in y_resampled]

            # Update data
            self.X = pd.DataFrame(X_resampled, columns=self.X.columns)
            self.y = pd.Series(y_original, name=self.y.name)

            logger.info("Data augmentation applied successfully.")

        except ImportError:
            logger.error("imbalanced-learn not available. Install with: pip install imbalanced-learn")
            raise

    def handle_class_imbalance(self, method: str = 'smote', sampling_strategy: str = 'auto') -> None:
        """
        Handle class imbalance in the dataset.

        Args:
            method: Imbalance handling method ('smote', 'random_oversample', 'random_undersample', 'class_weights')
            sampling_strategy: Sampling strategy for resampling methods
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be preprocessed before handling class imbalance.")

        try:
            if method == 'smote':
                self._apply_smote(sampling_strategy)
            elif method == 'random_oversample':
                self._apply_random_oversample(sampling_strategy)
            elif method == 'random_undersample':
                self._apply_random_undersample(sampling_strategy)
            elif method == 'class_weights':
                self._calculate_class_weights()
            else:
                raise ValueError(f"Unsupported imbalance handling method: {method}")
            logger.info(f"Class imbalance handled using {method}")
        except Exception as e:
            logger.error(f"Class imbalance handling failed: {str(e)}")
            raise

    def _apply_smote(self, sampling_strategy: str) -> None:
        """Apply SMOTE oversampling."""
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(self.X, self.y)

            self.X = pd.DataFrame(X_resampled, columns=self.X.columns)
            self.y = pd.Series(y_resampled, name=self.y.name)

            logger.info(f"SMOTE applied. New dataset size: {len(self.X)}")

        except ImportError:
            logger.error("imbalanced-learn not available. Install with: pip install imbalanced-learn")
            raise

    def _apply_random_oversample(self, sampling_strategy: str) -> None:
        """Apply random oversampling."""
        try:
            from imblearn.over_sampling import RandomOverSampler

            oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = oversampler.fit_resample(self.X, self.y)

            self.X = pd.DataFrame(X_resampled, columns=self.X.columns)
            self.y = pd.Series(y_resampled, name=self.y.name)

            logger.info(f"Random oversampling applied. New dataset size: {len(self.X)}")

        except ImportError:
            logger.error("imbalanced-learn not available. Install with: pip install imbalanced-learn")
            raise

    def _apply_random_undersample(self, sampling_strategy: str) -> None:
        """Apply random undersampling."""
        try:
            from imblearn.under_sampling import RandomUnderSampler

            undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(self.X, self.y)

            self.X = pd.DataFrame(X_resampled, columns=self.X.columns)
            self.y = pd.Series(y_resampled, name=self.y.name)

            logger.info(f"Random undersampling applied. New dataset size: {len(self.X)}")

        except ImportError:
            logger.error("imbalanced-learn not available. Install with: pip install imbalanced-learn")
            raise

    def _calculate_class_weights(self) -> None:
        """Calculate class weights for imbalanced datasets."""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(self.y)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y)

        # Store as sample weights
        weight_dict = dict(zip(classes, class_weights))
        self.sample_weights = np.array([weight_dict[label] for label in self.y])

        logger.info("Class weights calculated for imbalanced dataset.")

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

    def reduce_dimensionality(self, n_components: int = 2, method: str = 'pca') -> None:
        """
        Reduces dimensionality of the feature set.
        
        Args:
            n_components (int): Number of components to keep.
            method (str): Method to use ('pca' or 'tsne').
        
        Raises:
            ValueError: If method is unsupported or data is not preprocessed.
        """
        if self.X is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        if method == 'pca':
            pca = PCA(n_components=n_components)
            self.X = pd.DataFrame(pca.fit_transform(self.X), columns=[f'PC{i+1}' for i in range(n_components)], index=self.X.index)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=n_components)
            self.X = pd.DataFrame(tsne.fit_transform(self.X), columns=[f'TSNE{i+1}' for i in range(n_components)], index=self.X.index)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        logger.info(f"Dimensionality reduced using {method} to {n_components} components.")
