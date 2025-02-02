"""
Augini: AI-powered data analysis and feature engineering.
"""

from .data_analyzer import DataAnalyzer
from .data_engineer import DataEngineer, FeatureSpec
from .config import AuginiConfig

__version__ = "0.1.0"

__all__ = [
    "DataAnalyzer",
    "DataEngineer",
    "FeatureSpec",
    "AuginiConfig",
]
