from setuptools import setup, find_packages

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
        'joblib'
    ],
    author='Michael Kupermann',
    description='An advanced framework for equitable and responsible machine learning with support for fairness, interpretability, and robustness.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mkupermann/EquiML',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
