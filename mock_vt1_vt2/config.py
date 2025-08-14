"""
Configuration settings for the vt1-vt2 prediction project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data files
RAW_DATA_FILE = DATA_RAW / "raw.csv"
INTERIM_FEATURES_FILE = DATA_INTERIM / "features.csv"
INTERIM_TARGETS_FILE = DATA_INTERIM / "targets.csv"
INTERIM_DATA_FILE = DATA_INTERIM / "interim_data.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering parameters
TIME_SERIES_WINDOW_SIZE = 100  # For rolling window features
FFT_N_COMPONENTS = 10  # Number of FFT components to keep
STATISTICAL_FEATURES = [
    'mean', 'std', 'min', 'max', 'median', 
    'skewness', 'kurtosis', 'q25', 'q75'
]

# Model hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# EMBEDDED-FRIENDLY MODEL PARAMETERS
EMBEDDED_XGBOOST_PARAMS = {
    'n_estimators': 20,  # Reduced from 100
    'max_depth': 4,      # Reduced from 6
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': 1          # Single thread for embedded
}

EMBEDDED_RF_PARAMS = {
    'n_estimators': 10,  # Very small forest
    'max_depth': 5,      # Limited depth
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}

EMBEDDED_LINEAR_PARAMS = {
    'fit_intercept': True,
    'copy_X': True,
    'n_jobs': 1
}

# LSTM parameters (for comparison - not recommended for embedded)
LSTM_PARAMS = {
    'units': 50,
    'dropout': 0.2,
    'recurrent_dropout': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2
}

# Ensure directories exist
for directory in [DATA_PROCESSED, DATA_INTERIM, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
