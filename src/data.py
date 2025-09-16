import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy import stats
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import cv2  # For image data support
import logging
from typing import List, Optional, Tuple, Union

nltk.download('punkt')
nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, dataset_path: Optional[str] = None, sensitive_features: Optional[List[str]] = None, text_features: Optional[List[str]] = None, image_features: Optional[List[str]] = None, stopword_language: str = 'english'):
        """
        Initializes the Data object.
        
        Args:
            dataset_path (str, optional): Path to the dataset file.
            sensitive_features (list, optional): List of sensitive feature names.
            text_features (list, optional): List of text feature names for processing.
            image_features (list, optional): List of image feature names or paths for processing.
            stopword_language (str): The language for stopwords (default: 'english').
        """
        self.dataset_path = dataset_path
        self.sensitive_features = sensitive_features or []
        self.text_features = text_features or []
        self.image_features = image_features or []
        self.stopword_language = stopword_language
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, dataset_path: Optional[str] = None) -> None:
        """
        Loads the dataset from the specified path. Supports CSV, JSON, and Excel files.
        
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
            else:
                raise ValueError("Unsupported file format. Please use CSV, JSON, or Excel files.")
            logger.info(f"Data loaded successfully from {self.dataset_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def preprocess(self, target_column: str, numerical_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, impute_strategy: str = 'mean', scaling: str = 'standard', max_text_features: Optional[int] = 100) -> None:
        """
        Preprocesses the data by handling missing values, encoding categorical variables,
        scaling numerical features, processing text features, and handling image data.
        
        Args:
            target_column (str): The name of the target variable column.
            numerical_features (list, optional): List of numerical feature names.
            categorical_features (list, optional): List of categorical feature names.
            impute_strategy (str): Imputation strategy for numerical features ('mean', 'median', 'constant').
            scaling (str): Scaling method for numerical features ('standard', 'minmax', 'robust').
            max_text_features (int, optional): The maximum number of features for TF-IDF vectorization.
        
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
                vectorizer = TfidfVectorizer(max_features=max_text_features)
                text_features_array = vectorizer.fit_transform(self.X[feature]).toarray()
                text_df = pd.DataFrame(text_features_array, columns=[f"{feature}_{i}" for i in range(text_features_array.shape[1])], index=self.X.index)
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
            text = str(text).lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in stopwords.words(self.stopword_language)]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            raise ValueError(f"Text cleaning failed: {str(e)}")

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
                    raise ValueError(f"Image at path '{path}' could not be read.")
            except Exception as e:
                logger.warning(f"Failed to process image '{path}': {str(e)}")
                raise ValueError(f"Failed to process image '{path}': {str(e)}")
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
