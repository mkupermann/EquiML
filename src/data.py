import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy import stats
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import cv2  # For image data support

nltk.download('punkt')
nltk.download('stopwords')

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
    
    def __init__(self, dataset_path=None, sensitive_features=None, text_features=None, image_features=None):
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

    def preprocess(self, target_column, numerical_features=None, categorical_features=None):
        """
        Preprocesses the data by handling missing values, encoding categorical variables,
        scaling numerical features, processing text features, and handling image data.
        
        Args:
            target_column (str): The name of the target variable column.
            numerical_features (list, optional): List of numerical feature names.
            categorical_features (list, optional): List of categorical feature names.
        """
        if not self.df:
            raise ValueError("No data loaded. Call load_data() first.")
        self.y = self.df[target_column]
        self.X = self.df.drop(columns=[target_column])
        
        # Handle missing values
        if numerical_features:
            num_imputer = SimpleImputer(strategy='mean')
            self.X[numerical_features] = num_imputer.fit_transform(self.X[numerical_features])
        if categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.X[categorical_features] = cat_imputer.fit_transform(self.X[categorical_features])
        
        # Process text features
        if self.text_features:
            for feature in self.text_features:
                self.X[feature] = self.X[feature].apply(self._clean_text)
                vectorizer = TfidfVectorizer(max_features=100)
                text_features = vectorizer.fit_transform(self.X[feature]).toarray()
                text_df = pd.DataFrame(text_features, columns=[f"{feature}_{i}" for i in range(100)])
                self.X = pd.concat([self.X.drop(columns=[feature]), text_df], axis=1)
        
        # Process image features (assuming paths to images are provided)
        if self.image_features:
            for feature in self.image_features:
                image_data = self._process_images(self.X[feature])
                image_df = pd.DataFrame(image_data, columns=[f"{feature}_pixel_{i}" for i in range(image_data.shape[1])])
                self.X = pd.concat([self.X.drop(columns=[feature]), image_df], axis=1)
        
        # Encode categorical features
        if categorical_features:
            encoder = OneHotEncoder(drop='first', sparse_output=False)
            encoded_cats = encoder.fit_transform(self.X[categorical_features])
            cat_columns = encoder.get_feature_names_out(categorical_features)
            self.X = pd.concat([self.X.drop(columns=categorical_features),
                               pd.DataFrame(encoded_cats, columns=cat_columns)], axis=1)
        
        # Scale numerical features
        if numerical_features:
            scaler = StandardScaler()
            self.X[numerical_features] = scaler.fit_transform(self.X[numerical_features])

    def _clean_text(self, text):
        """
        Helper function to clean text data.
        
        Args:
            text (str): Raw text input.
        
        Returns:
            str: Cleaned text.
        """
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)

    def _process_images(self, image_paths, target_size=(32, 32)):
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
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Grayscale for simplicity
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                processed_images.append(img_resized.flatten())
            else:
                processed_images.append(np.zeros(target_size[0] * target_size[1]))
        return np.array(processed_images)

    def detect_outliers(self, features=None, method='zscore', threshold=3, action='flag'):
        """
        Detects and optionally handles outliers in the specified features.
        
        Args:
            features (list, optional): List of features to check for outliers. Defaults to all numerical features.
            method (str): Method to use ('zscore' or 'iqr').
            threshold (float): Threshold for detecting outliers.
            action (str): Action to take ('flag', 'remove', 'cap').
        
        Returns:
            pd.DataFrame: DataFrame indicating outliers (if action='flag').
        """
        if features is None:
            features = self.X.select_dtypes(include=[np.number]).columns
        outliers = pd.DataFrame(index=self.X.index)
        
        for feature in features:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(self.X[feature]))
                outliers[feature] = z_scores > threshold
            elif method == 'iqr':
                Q1 = self.X[feature].quantile(0.25)
                Q3 = self.X[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers[feature] = (self.X[feature] < (Q1 - 1.5 * IQR)) | (self.X[feature] > (Q3 + 1.5 * IQR))
        
        if action == 'remove':
            self.X = self.X[~outliers.any(axis=1)]
            self.y = self.y[~outliers.any(axis=1)]
        elif action == 'cap':
            for feature in features:
                if method == 'zscore':
                    self.X[feature] = self.X[feature].clip(-threshold, threshold)
                elif method == 'iqr':
                    Q1 = self.X[feature].quantile(0.25)
                    Q3 = self.X[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    self.X[feature] = self.X[feature].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        elif action == 'flag':
            return outliers

    def feature_engineering(self, polynomial_degree=2, interaction_only=False, include_log=False):
        """
        Performs feature engineering by creating polynomial features, interaction terms, and optional log transformations.
        
        Args:
            polynomial_degree (int): Degree of polynomial features.
            interaction_only (bool): If True, only interaction features are created.
            include_log (bool): If True, adds log-transformed features for numerical columns.
        """
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        poly = PolynomialFeatures(degree=polynomial_degree, interaction_only=interaction_only)
        poly_features = poly.fit_transform(self.X[numerical_cols])
        poly_columns = poly.get_feature_names_out(numerical_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_columns, index=self.X.index)
        
        # Add log transformations if requested
        if include_log:
            for col in numerical_cols:
                if (self.X[col] > 0).all():  # Ensure no non-positive values
                    poly_df[f'log_{col}'] = np.log1p(self.X[col])
        
        # Combine with original non-numerical features
        self.X = pd.concat([self.X.drop(columns=numerical_cols), poly_df], axis=1)

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def reduce_dimensionality(self, n_components=2, method='pca'):
        """
        Reduces dimensionality of the feature set.
        
        Args:
            n_components (int): Number of components to keep.
            method (str): Method to use ('pca' or 'tsne').
        """
        if method == 'pca':
            pca = PCA(n_components=n_components)
            self.X = pd.DataFrame(pca.fit_transform(self.X), columns=[f'PC{i+1}' for i in range(n_components)])
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=n_components)
            self.X = pd.DataFrame(tsne.fit_transform(self.X), columns=[f'TSNE{i+1}' for i in range(n_components)])
