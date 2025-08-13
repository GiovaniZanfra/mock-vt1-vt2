"""
VT1/VT2 Prediction Pipeline Package

A comprehensive machine learning pipeline for predicting VT1 and VT2
from cycling data using feature engineering, data augmentation, and
multiple machine learning algorithms.
"""

from .pipeline import VTPredictionPipeline
from .feature_engineering import FeatureEngineer
from .data_augmentation import DataAugmenter
from .preprocessing import DataPreprocessor
from .model_trainer import ModelTrainer
from .evaluation import ModelEvaluator

__version__ = "1.0.0"
__author__ = "VT1/VT2 Prediction Team"

__all__ = [
    'VTPredictionPipeline',
    'FeatureEngineer',
    'DataAugmenter',
    'DataPreprocessor',
    'ModelTrainer',
    'ModelEvaluator'
]
