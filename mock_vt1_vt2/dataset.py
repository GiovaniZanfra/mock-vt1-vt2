"""
Data processing pipeline for vt1 and vt2 prediction.
Moves data from raw to interim following cookie cutter data science rules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data() -> pd.DataFrame:
    """
    Load raw data from data/raw/raw.csv
    """
    raw_data_path = Path("data/raw/raw.csv")
    logger.info(f"Loading raw data from {raw_data_path}")
    
    try:
        # Try to read with different encodings and separators
        df = pd.read_csv(raw_data_path)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def explore_data_structure(df: pd.DataFrame) -> None:
    """
    Explore the structure of the raw data
    """
    logger.info("=== Data Structure Exploration ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    
    # Check for vt1 and vt2 columns
    vt_columns = [col for col in df.columns if 'vt1' in col.lower() or 'vt2' in col.lower()]
    if vt_columns:
        logger.info(f"Found VT columns: {vt_columns}")
    else:
        logger.info("No VT columns found - they might be target variables to predict")
    
    # Show first few rows
    logger.info(f"First 5 rows:\n{df.head()}")

def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the raw data and split into features and targets
    """
    logger.info("Processing data...")
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Parse string representations of lists in hr and speed columns
    import ast
    
    # Parse hr column if it exists
    if 'hr' in df_processed.columns:
        logger.info("Parsing hr column...")
        try:
            df_processed['hr'] = df_processed['hr'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        except Exception as e:
            logger.warning(f"Could not parse hr column: {e}")
    
    # Parse speed column if it exists
    if 'speed' in df_processed.columns:
        logger.info("Parsing speed column...")
        try:
            df_processed['speed'] = df_processed['speed'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        except Exception as e:
            logger.warning(f"Could not parse speed column: {e}")
    
    # Check if vt1 and vt2 are in the data
    vt1_col = None
    vt2_col = None
    
    for col in df_processed.columns:
        if 'vt1' in col.lower():
            vt1_col = col
        elif 'vt2' in col.lower():
            vt2_col = col
    
    # If vt1 and vt2 are not found, we'll need to create them or they're targets
    if vt1_col is None or vt2_col is None:
        logger.info("VT1 and VT2 not found in data - assuming they are target variables")
        # For now, let's assume the last two columns are targets
        # This will need to be adjusted based on actual data structure
        if len(df_processed.columns) >= 2:
            vt1_col = df_processed.columns[-2]
            vt2_col = df_processed.columns[-1]
            logger.info(f"Using {vt1_col} and {vt2_col} as target variables")
    
    # Separate features and targets
    if vt1_col and vt2_col:
        target_cols = [vt1_col, vt2_col]
        feature_cols = [col for col in df_processed.columns if col not in target_cols]
        
        X = df_processed[feature_cols]
        y = df_processed[target_cols]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Targets shape: {y.shape}")
        
        return X, y
    else:
        logger.warning("Could not identify target columns")
        return df_processed, pd.DataFrame()

def save_interim_data(X: pd.DataFrame, y: pd.DataFrame) -> None:
    """
    Save interim data to data/interim/
    """
    interim_dir = Path("data/interim")
    interim_dir.mkdir(exist_ok=True)
    
    # Save features
    X.to_csv(interim_dir / "features.csv", index=False)
    logger.info(f"Saved features to {interim_dir / 'features.csv'}")
    
    # Save targets
    if not y.empty:
        y.to_csv(interim_dir / "targets.csv", index=False)
        logger.info(f"Saved targets to {interim_dir / 'targets.csv'}")
    
    # Save combined data
    if not y.empty:
        combined = pd.concat([X, y], axis=1)
        combined.to_csv(interim_dir / "interim_data.csv", index=False)
        logger.info(f"Saved combined data to {interim_dir / 'interim_data.csv'}")

def main():
    """
    Main data processing pipeline
    """
    logger.info("Starting data processing pipeline...")
    
    # Load raw data
    df = load_raw_data()
    
    # Explore data structure
    explore_data_structure(df)
    
    # Process data
    X, y = process_data(df)
    
    # Save interim data
    save_interim_data(X, y)
    
    logger.info("Data processing pipeline completed!")

if __name__ == "__main__":
    main()
