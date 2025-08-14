"""
Model prediction for vt1 and vt2.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from ..config import *
from ..features import TimeSeriesFeatureExtractor

logger = logging.getLogger(__name__)

class VT1VT2Predictor:
    """
    Class for making predictions using trained models.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.models = {}
        self.scaler = None
        self.feature_extractor = TimeSeriesFeatureExtractor()
        
    def load_models(self) -> None:
        """
        Load trained models and scaler.
        """
        models_dir = MODELS_DIR
        
        # Load scaler
        scaler_path = models_dir / f"scaler_{self.model_type}.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        # Load models
        model_files = list(models_dir.glob(f"*_{self.model_type}_model.pkl"))
        for model_file in model_files:
            target_name = model_file.stem.replace(f"_{self.model_type}_model", "")
            self.models[target_name] = joblib.load(model_file)
            logger.info(f"Loaded model for {target_name} from {model_file}")
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from raw data.
        """
        logger.info("Extracting features from data...")
        
        # Extract features for each column
        all_features = []
        
        for col in data.columns:
            logger.info(f"Extracting features from column: {col}")
            features = self.feature_extractor.extract_all_features(data[col])
            features['column_name'] = col
            all_features.append(features)
        
        # Create features dataframe
        features_df = pd.DataFrame(all_features)
        
        # Aggregate features across columns
        if len(features_df) > 0:
            feature_cols = [col for col in features_df.columns if col != 'column_name']
            aggregated_features = features_df[feature_cols].mean()
            final_features = pd.DataFrame([aggregated_features])
        else:
            final_features = pd.DataFrame()
        
        logger.info(f"Extracted features with shape: {final_features.shape}")
        return final_features
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        """
        # Load models if not already loaded
        if not self.models:
            self.load_models()
        
        # Extract features
        features_df = self.extract_features(data)
        
        if features_df.empty:
            raise ValueError("No features extracted from data")
        
        # Scale features
        X_scaled = self.scaler.transform(features_df)
        
        # Make predictions
        predictions = {}
        for target_name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[target_name] = pred
        
        return pd.DataFrame(predictions)
    
    def predict_from_raw_file(self, raw_file_path: Path) -> pd.DataFrame:
        """
        Make predictions from a raw data file.
        """
        logger.info(f"Loading raw data from {raw_file_path}")
        
        # Load raw data
        data = pd.read_csv(raw_file_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Make predictions
        predictions = self.predict(data)
        
        return predictions

def main():
    """
    Main prediction pipeline.
    """
    logger.info("Starting prediction pipeline...")
    
    # Initialize predictor
    predictor = VT1VT2Predictor(model_type='xgboost')
    
    try:
        # Make predictions on raw data
        predictions = predictor.predict_from_raw_file(RAW_DATA_FILE)
        
        # Save predictions
        predictions_path = DATA_INTERIM / "predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")
        
        # Print predictions
        logger.info("Predictions:")
        for col in predictions.columns:
            logger.info(f"{col}: {predictions[col].iloc[0]:.4f}")
            
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error("Make sure models are trained first by running train.py")

if __name__ == "__main__":
    main()
