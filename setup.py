from setuptools import setup, find_packages

setup(
    name='equiml',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',  # For machine learning models and utilities
        'fairlearn',     # For fairness metrics and mitigation
        'shap',          # For model interpretability
        'matplotlib',    # For plotting and visualization
        'seaborn',       # For enhanced visualization
        'streamlit',     # For interactive web apps
        'pandas',        # For data manipulation
        'numpy',         # For numerical operations
        'scipy',         # For statistical tests (e.g., data drift detection)
        'joblib',        # For model serialization (saving/loading models)
    ],
    author='Michael Kupermann',
    description='A framework for equitable and responsible machine learning',
)
