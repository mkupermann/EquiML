from setuptools import setup, find_packages

setup(
    name='equiml',
    version='0.1.0',
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
        
    ],
    author='Michael Kupermann',
    description='A framework for equitable and responsible machine learning',
)
