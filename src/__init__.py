"""
EquiML: A Framework for Equitable and Responsible Machine Learning

EquiML provides comprehensive tools for building fair, transparent, and accountable
machine learning models with built-in bias detection, mitigation, and monitoring.
"""

__version__ = "0.2.0"
__author__ = "Michael Kupermann"
__email__ = "mkupermann@kupermann.com"

from .data import Data
from .model import Model
from .evaluation import EquiMLEvaluation
from .monitoring import BiasMonitor, DriftDetector

__all__ = [
    "Data",
    "Model",
    "EquiMLEvaluation",
    "BiasMonitor",
    "DriftDetector"
]