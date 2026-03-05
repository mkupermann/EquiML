"""
EquiML - Quick fairness audits for ML datasets.

Wraps fairlearn, SHAP, and scikit-learn into a single audit pipeline.
"""

__version__ = "1.0.0"

from .data import Data
from .model import Model
from .evaluation import EquiMLEvaluation
from .monitoring import BiasMonitor, DriftDetector

__all__ = [
    "Data",
    "Model",
    "EquiMLEvaluation",
    "BiasMonitor",
    "DriftDetector",
]
