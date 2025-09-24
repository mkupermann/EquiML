from setuptools import setup, find_packages
import os

def _get_long_description():
    """Get long description, handling Docker build context"""
    readme_path = 'README.md'
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            pass
    return 'EquiML: A comprehensive framework for equitable and responsible machine learning with support for fairness, interpretability, and robustness.'

setup(
    name='equiml',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'fairlearn',
        'shap',
        'matplotlib',
        'seaborn',
        'streamlit',
        'pandas',
        'numpy',
        'xgboost',
        'optuna',
        'skl2onnx',
        'onnx',
        'plotly',
        'statsmodels',
        'lime',
        'scipy',
        'joblib',
        'lightgbm'
    ],
    extras_require={
        'image': ['opencv-python>=4.5.5.64'],
        'text': ['nltk>=3.7'],
        'dev': ['pytest>=7.1.1', 'flake8>=4.0.1', 'sphinx>=4.4.0', 'coverage>=6.3.2'],
        'deploy': ['docker>=5.0.3']
    },
    author='Michael Kupermann',
    description='An advanced framework for equitable and responsible machine learning with support for fairness, interpretability, and robustness.',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/mkupermann/EquiML',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'equiml-app = src.streamlit_app:main',
        ],
    },
    include_package_data=True,
)
