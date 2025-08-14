"""
Model training for vt1 and vt2 prediction.
Optimized for embedded systems.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from ..config import *
from ..features import EmbeddedTimeSeriesFeatureExtractor

logger = logging.getLogger(__name__)

class EmbeddedVT1VT2Predictor:
    """
    Lightweight predictor optimized for embedded systems.
    """
    
    def __init__(self, model_type: str = 'linear'):
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_extractor = EmbeddedTimeSeriesFeatureExtractor(use_fft=False)
        
        # Model size limits for embedded systems
        self.max_model_size_mb = 1.0  # 1MB limit for embedded
        
    def get_model_params(self, model_type: str) -> dict:
        """
        Get parameters for embedded-optimized models.
        """
        if model_type == 'linear':
            return EMBEDDED_LINEAR_PARAMS
        elif model_type == 'decision_tree':
            return {
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': RANDOM_STATE
            }
        elif model_type == 'random_forest':
            return EMBEDDED_RF_PARAMS
        elif model_type == 'xgboost':
            return EMBEDDED_XGBOOST_PARAMS
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_model(self, model_type: str):
        """
        Create a model instance with embedded-optimized parameters.
        """
        params = self.get_model_params(model_type)
        
        if model_type == 'linear':
            return LinearRegression(**params)
        elif model_type == 'decision_tree':
            return DecisionTreeRegressor(**params)
        elif model_type == 'random_forest':
            return RandomForestRegressor(**params)
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def estimate_model_size(self, model) -> float:
        """
        Estimate model size in MB.
        """
        try:
            # Save model to bytes and estimate size
            import io
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            size_bytes = buffer.tell()
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        except:
            return 0.1  # Default estimate
    
    def prepare_data(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(features_df)
        
        # Prepare targets
        y = targets_df.values
        
        return X_scaled, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, target_name: str) -> Any:
        """
        Train a model for a specific target.
        """
        logger.info(f"Training {self.model_type} model for {target_name}")
        
        model = self.create_model(self.model_type)
        
        # Train model
        model.fit(X, y)
        
        # Check model size
        model_size = self.estimate_model_size(model)
        logger.info(f"Model size for {target_name}: {model_size:.2f} MB")
        
        if model_size > self.max_model_size_mb:
            logger.warning(f"Model size ({model_size:.2f} MB) exceeds limit ({self.max_model_size_mb} MB)")
            logger.info("Consider using a simpler model type")
        
        return model
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, target_name: str) -> Dict[str, float]:
        """
        Evaluate model performance.
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Model performance for {target_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def compare_models(self, X: np.ndarray, y: np.ndarray, target_name: str) -> Dict[str, Dict]:
        """
        Compare different model types for embedded suitability.
        """
        model_types = ['linear', 'decision_tree', 'random_forest', 'xgboost']
        results = {}
        
        logger.info(f"Comparing models for {target_name}...")
        
        for model_type in model_types:
            logger.info(f"Testing {model_type}...")
            
            # Create and train model
            model = self.create_model(model_type)
            model.fit(X, y)
            
            # Evaluate
            metrics = self.evaluate_model(model, X, y, target_name)
            
            # Estimate size
            model_size = self.estimate_model_size(model)
            
            results[model_type] = {
                'metrics': metrics,
                'size_mb': model_size,
                'model': model
            }
            
            logger.info(f"{model_type}: R²={metrics['r2']:.4f}, Size={model_size:.2f}MB")
        
        return results
    
    def train_all_models(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train models for both vt1 and vt2.
        """
        logger.info("Training models for vt1 and vt2...")
        
        # Prepare data
        X, y = self.prepare_data(features_df, targets_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        results = {}
        
        # Train model for each target
        for i, target_name in enumerate(targets_df.columns):
            logger.info(f"Training model for {target_name}")
            
            # Compare models first
            comparison_results = self.compare_models(X_train, y_train[:, i], target_name)
            
            # Choose best model based on R² and size
            best_model_type = None
            best_score = -np.inf
            
            for model_type, result in comparison_results.items():
                score = result['metrics']['r2'] - 0.1 * result['size_mb']  # Penalize large models
                if score > best_score:
                    best_score = score
                    best_model_type = model_type
            
            logger.info(f"Best model for {target_name}: {best_model_type}")
            
            # Use the best model
            self.models[target_name] = comparison_results[best_model_type]['model']
            results[target_name] = comparison_results[best_model_type]['metrics']
        
        # Save models and scaler
        self.save_models()
        
        return results
    
    def save_models(self) -> None:
        """
        Save trained models and scaler.
        """
        models_dir = MODELS_DIR
        models_dir.mkdir(exist_ok=True)
        
        # Save models
        for target_name, model in self.models.items():
            model_path = models_dir / f"{target_name}_embedded_model.pkl"
            joblib.dump(model, model_path)
            
            # Log model size
            model_size = self.estimate_model_size(model)
            logger.info(f"Saved model for {target_name} to {model_path} (Size: {model_size:.2f} MB)")
        
        # Save scaler
        scaler_path = models_dir / "embedded_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using trained models.
        """
        # Scale features
        X_scaled = self.scaler.transform(features_df)
        
        # Make predictions
        predictions = {}
        for target_name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[target_name] = pred
        
        return pd.DataFrame(predictions)

def main():
    """
    Main training pipeline.
    """
    logger.info("Starting model training pipeline...")
    
    # Load features and targets
    try:
        features_df = pd.read_csv(DATA_INTERIM / "engineered_features.csv")
        targets_df = pd.read_csv(INTERIM_TARGETS_FILE)
        logger.info(f"Loaded features with shape: {features_df.shape}")
        logger.info(f"Loaded targets with shape: {targets_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"Required data not found: {e}")
        logger.error("Please run features.py first to create features.")
        return
    
    # Initialize embedded predictor
    predictor = EmbeddedVT1VT2Predictor(model_type='linear')
    
    # Train models
    results = predictor.train_all_models(features_df, targets_df)
    
    # Print summary
    logger.info("Training completed!")
    logger.info("Model performance summary:")
    for target, metrics in results.items():
        logger.info(f"{target}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")

if __name__ == "__main__":
    main()
