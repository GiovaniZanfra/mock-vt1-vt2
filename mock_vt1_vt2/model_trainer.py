"""
Modeling Module for VT1/VT2 Prediction

This module handles model training, hyperparameter tuning, and model selection
for VT1 and VT2 prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Try to import XGBoost and LightGBM, but handle case where they're not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model training class for VT1/VT2 prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model trainer with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}
        
    def train_all_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Train all specified models for VT1 and VT2 prediction.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            
        Returns:
            Dictionary of trained models
        """
        self.logger.info("Starting model training...")
        
        targets = self.config['targets']
        algorithms = self.config['algorithms']
        
        # Separate features and targets
        feature_cols = [col for col in train_data.columns if col not in targets + ['idx', 'sid']]
        
        X_train = train_data[feature_cols]
        y_train = train_data[targets]
        X_test = test_data[feature_cols]
        y_test = test_data[targets]
        
        # Train models for each target
        for target in targets:
            self.logger.info(f"Training models for {target}...")
            
            y_train_target = y_train[target]
            y_test_target = y_test[target]
            
            target_models = {}
            
            for algorithm in algorithms:
                try:
                    model = self._train_single_model(
                        algorithm, X_train, y_train_target, X_test, y_test_target, target
                    )
                    target_models[algorithm] = model
                    self.logger.info(f"Successfully trained {algorithm} for {target}")
                except Exception as e:
                    self.logger.error(f"Failed to train {algorithm} for {target}: {str(e)}")
                    continue
            
            self.models[target] = target_models
            
            # Select best model for this target
            if target_models:
                best_model_name = self._select_best_model(target_models, X_test, y_test_target, target)
                self.best_models[target] = target_models[best_model_name]
                self.logger.info(f"Best model for {target}: {best_model_name}")
        
        return self.best_models
    
    def _train_single_model(self, algorithm: str, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series, target: str) -> Dict:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            algorithm: Algorithm name
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            target: Target variable name
            
        Returns:
            Dictionary containing model and metadata
        """
        # Get model configuration
        model_config = self._get_model_config(algorithm)
        
        # Perform cross-validation
        cv_scores = self._cross_validate_model(model_config, X_train, y_train)
        
        # Hyperparameter tuning if enabled
        if self.config['hyperparameter_tuning']['enabled']:
            best_params = self._tune_hyperparameters(algorithm, X_train, y_train)
            model_config.update(best_params)
        
        # Train final model
        final_model = self._create_model(algorithm, model_config)
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = final_model.predict(X_test)
        test_metrics = self._calculate_metrics(y_test, y_pred)
        
        return {
            'model': final_model,
            'algorithm': algorithm,
            'target': target,
            'cv_scores': cv_scores,
            'test_metrics': test_metrics,
            'hyperparameters': model_config
        }
    
    def _get_model_config(self, algorithm: str) -> Dict:
        """Get default configuration for a model."""
        configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'elastic_net': {
                'alpha': 0.1,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'random_state': 42
            },
            'svr': {
                'C': 1.0,
                'epsilon': 0.1,
                'kernel': 'rbf',
                'gamma': 'scale'
            }
        }
        
        return configs.get(algorithm, {})
    
    def _create_model(self, algorithm: str, config: Dict):
        """Create a model instance."""
        if algorithm == 'random_forest':
            return RandomForestRegressor(**config)
        elif algorithm == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            return xgb.XGBRegressor(**config)
        elif algorithm == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            return lgb.LGBMRegressor(**config)
        elif algorithm == 'elastic_net':
            return ElasticNet(**config)
        elif algorithm == 'svr':
            return SVR(**config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _cross_validate_model(self, model_config: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform cross-validation."""
        cv_config = self.config['cv']
        n_splits = cv_config['n_splits']
        random_state = cv_config['random_state']
        
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Create model
        model = self._create_model('random_forest', model_config)  # Default to RF for CV
        
        # Calculate CV scores
        mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        return {
            'mae_mean': -mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'mse_mean': -mse_scores.mean(),
            'mse_std': mse_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
    
    def _tune_hyperparameters(self, algorithm: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform hyperparameter tuning."""
        self.logger.info(f"Tuning hyperparameters for {algorithm}...")
        
        tuning_config = self.config['hyperparameter_tuning']
        method = tuning_config['method']
        n_iterations = tuning_config['n_iterations']
        
        if method == 'random_search':
            return self._random_search_tuning(algorithm, X, y, n_iterations)
        else:
            self.logger.warning(f"Hyperparameter tuning method {method} not implemented")
            return {}
    
    def _random_search_tuning(self, algorithm: str, X: pd.DataFrame, y: pd.Series, n_iterations: int) -> Dict:
        """Perform random search hyperparameter tuning."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'elastic_net': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'svr': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'epsilon': [0.01, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        }
        
        param_grid = param_grids.get(algorithm, {})
        if not param_grid:
            return {}
        
        best_score = float('inf')
        best_params = {}
        
        for _ in range(n_iterations):
            # Randomly sample parameters
            params = {}
            for param, values in param_grid.items():
                params[param] = np.random.choice(values)
            
            # Add default parameters
            default_config = self._get_model_config(algorithm)
            for key, value in default_config.items():
                if key not in params:
                    params[key] = value
            
            # Evaluate parameters
            try:
                cv_scores = self._cross_validate_model(params, X, y)
                score = cv_scores['mae_mean']  # Use MAE as optimization metric
                
                if score < best_score:
                    best_score = score
                    best_params = params
            except:
                continue
        
        self.logger.info(f"Best parameters for {algorithm}: {best_params}")
        return best_params
    
    def _select_best_model(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series, target: str) -> str:
        """Select the best model based on test performance."""
        best_score = float('inf')
        best_model_name = None
        
        for model_name, model_info in models.items():
            score = model_info['test_metrics']['mae']
            if score < best_score:
                best_score = score
                best_model_name = model_name
        
        return best_model_name
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def save_models(self, output_path: str):
        """Save trained models to disk."""
        if not self.best_models:
            self.logger.warning("No models to save")
            return
        
        os.makedirs(output_path, exist_ok=True)
        
        for target, model_info in self.best_models.items():
            model_path = os.path.join(output_path, f"{target}_model.joblib")
            joblib.dump(model_info['model'], model_path)
            self.logger.info(f"Saved {target} model to {model_path}")
    
    def load_models(self, model_path: str):
        """Load trained models from disk."""
        targets = self.config['targets']
        
        for target in targets:
            model_file = os.path.join(model_path, f"{target}_model.joblib")
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                self.best_models[target] = {
                    'model': model,
                    'target': target
                }
                self.logger.info(f"Loaded {target} model from {model_file}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get a summary of all trained models."""
        summary_data = []
        
        for target, models in self.models.items():
            for algorithm, model_info in models.items():
                summary_data.append({
                    'target': target,
                    'algorithm': algorithm,
                    'mae': model_info['test_metrics']['mae'],
                    'rmse': model_info['test_metrics']['rmse'],
                    'r2': model_info['test_metrics']['r2'],
                    'mape': model_info['test_metrics']['mape'],
                    'cv_mae_mean': model_info['cv_scores']['mae_mean'],
                    'cv_r2_mean': model_info['cv_scores']['r2_mean']
                })
        
        return pd.DataFrame(summary_data)
