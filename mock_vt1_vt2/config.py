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

# External data source (for populating raw/)
# Allow overriding via environment variables to play with pipeline params
DATA_SOURCE_ROOT = Path(os.getenv("CRF_DATA_SOURCE_ROOT", "/home/g-brandao/workspace/crf-data"))
SOURCE_V3_SUBFOLDER = os.getenv("CRF_V3_SUBFOLDER", "v3")
SOURCE_METADATA_SUBFOLDER = os.getenv("CRF_METADATA_SUBFOLDER", "metadata")

# Helper to parse comma-separated environment variables

def _parse_list_env(var_name: str, default_list):
    value = os.getenv(var_name)
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(default_list)

# Protocols and candidate columns (tunable via env)
PROTOCOLS = _parse_list_env("CRF_PROTOCOLS", ["obk1"])  # Example default
HR_CANDIDATE_COLUMNS = _parse_list_env(
    "CRF_HR_CANDIDATES",
    [
        "cola-gw6_polar_hr",
        "cola-gw6_vo2max_hr",
        "cola-gw5_polar_hr",
        "cola-gw5_vo2max_hr",
    ],
)
SPEED_CANDIDATE_COLUMNS = _parse_list_env(
    "CRF_SPEED_CANDIDATES",
    [
        "cola-gw6_gps_speed",
        "cola-gw6_vo2max_gps-speed",
        "cola-gw5_gps_speed",
        "cola-gw5_vo2max_gps-speed",
    ],
)

# Metadata file stems that carry maximal info
MAXIMAL_FILENAMES = set(_parse_list_env("CRF_MAXIMAL_STEMS", ["maximal", "maximal_bk"]))

# Preferred raw parquet outputs
RAW_DATA_PARQUET_FILE = DATA_RAW / "raw.parquet"
MERGED_METADATA_PARQUET_FILE = DATA_RAW / "merged_metadata.parquet"

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
for directory in [DATA_PROCESSED, DATA_INTERIM, DATA_RAW, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
