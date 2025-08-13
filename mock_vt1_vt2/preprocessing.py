"""
Data Preprocessing Module for VT1/VT2 Prediction

This module handles data preprocessing tasks including:
- Feature scaling
- Feature selection
- Outlier detection and removal
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


class DataPreprocessor:
    """
    Data preprocessing class for VT1/VT2 prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize transformers
        self.scaler = None
        self.feature_selector = None
        self.outlier_detector = None
        self.selected_features = []
        self.is_fitted = False
        
    def fit_transform(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the preprocessor on training data and transform both train and test data.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            
        Returns:
            Tuple of (processed_train_data, processed_test_data)
        """
        self.logger.info("Starting data preprocessing...")
        
        # Separate features and targets
        target_cols = ['vt1', 'vt2']
        feature_cols = [col for col in train_data.columns if col not in target_cols + ['idx', 'sid']]
        
        # Only use numeric features for preprocessing
        numeric_cols = train_data[feature_cols].select_dtypes(include=[np.number]).columns
        non_numeric_cols = [col for col in feature_cols if col not in numeric_cols]
        
        X_train = train_data[numeric_cols]
        y_train = train_data[target_cols]
        X_test = test_data[numeric_cols]
        y_test = test_data[target_cols]
        
        # Apply outlier detection
        if self.config['outlier_detection']['enabled']:
            X_train, y_train = self._detect_and_remove_outliers(X_train, y_train)
        
        # Handle NaN values after outlier removal
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_test.median())
        
        # Apply feature selection
        if self.config['feature_selection']['enabled']:
            X_train, X_test = self._select_features(X_train, X_test, y_train)
        
        # Apply scaling
        X_train, X_test = self._scale_features(X_train, X_test)
        
        # Combine features and targets
        processed_train = pd.concat([X_train, y_train], axis=1)
        processed_test = pd.concat([X_test, y_test], axis=1)
        
        # Add back non-numeric columns
        if non_numeric_cols:
            processed_train = pd.concat([processed_train, train_data[non_numeric_cols]], axis=1)
            processed_test = pd.concat([processed_test, test_data[non_numeric_cols]], axis=1)
        
        # Add back index and subject ID columns
        if 'idx' in train_data.columns:
            processed_train['idx'] = train_data['idx'].iloc[:len(processed_train)]
        if 'sid' in train_data.columns:
            processed_train['sid'] = train_data['sid'].iloc[:len(processed_train)]
        
        if 'idx' in test_data.columns:
            processed_test['idx'] = test_data['idx'].iloc[:len(processed_test)]
        if 'sid' in test_data.columns:
            processed_test['sid'] = test_data['sid'].iloc[:len(processed_test)]
        
        self.is_fitted = True
        self.logger.info("Data preprocessing completed")
        
        return processed_train, processed_test
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        # Separate features and targets
        target_cols = ['vt1', 'vt2']
        feature_cols = [col for col in data.columns if col not in target_cols + ['idx', 'sid']]
        
        X = data[feature_cols]
        y = data[target_cols] if all(col in data.columns for col in target_cols) else None
        
        # Apply feature selection
        if self.selected_features:
            X = X[self.selected_features]
        
        # Apply scaling
        if self.scaler is not None:
            X = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        
        # Combine features and targets
        if y is not None:
            result = pd.concat([X, y], axis=1)
        else:
            result = X
        
        # Add back index and subject ID columns
        if 'idx' in data.columns:
            result['idx'] = data['idx']
        if 'sid' in data.columns:
            result['sid'] = data['sid']
        
        return result
    
    def _detect_and_remove_outliers(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect and remove outliers from the data.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            
        Returns:
            Tuple of (X_clean, y_clean)
        """
        self.logger.info("Detecting and removing outliers...")
        
        config = self.config['outlier_detection']
        method = config['method']
        contamination = config['contamination']
        
        if method == 'isolation_forest':
            detector = IsolationForest(
                contamination=contamination,
                random_state=42
            )
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=20
            )
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
        else:
            self.logger.warning(f"Unknown outlier detection method: {method}")
            return X, y
        
        # Fit and predict outliers
        outlier_labels = detector.fit_predict(X)
        
        # Remove outliers
        inlier_mask = outlier_labels == 1
        X_clean = X[inlier_mask]
        y_clean = y[inlier_mask]
        
        n_outliers = len(X) - len(X_clean)
        self.logger.info(f"Removed {n_outliers} outliers ({n_outliers/len(X)*100:.1f}%)")
        
        self.outlier_detector = detector
        return X_clean, y_clean
    
    def _select_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select the most important features.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            
        Returns:
            Tuple of (X_train_selected, X_test_selected)
        """
        self.logger.info("Selecting features...")
        
        config = self.config['feature_selection']
        method = config['method']
        n_features = config['n_features']
        
        if method == 'mutual_info':
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(X_train, y_train.iloc[:, 0])  # Use first target
            feature_scores = pd.Series(mi_scores, index=X_train.columns)
            selected_features = feature_scores.nlargest(n_features).index.tolist()
            
        elif method == 'f_regression':
            # Calculate F-statistics
            f_scores, _ = f_regression(X_train, y_train.iloc[:, 0])  # Use first target
            feature_scores = pd.Series(f_scores, index=X_train.columns)
            selected_features = feature_scores.nlargest(n_features).index.tolist()
            
        elif method == 'rfe':
            # Recursive feature elimination
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            rfe.fit(X_train, y_train.iloc[:, 0])  # Use first target
            selected_features = X_train.columns[rfe.support_].tolist()
            
        else:
            self.logger.warning(f"Unknown feature selection method: {method}")
            return X_train, X_test
        
        # Select features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        self.selected_features = selected_features
        self.logger.info(f"Selected {len(selected_features)} features")
        
        return X_train_selected, X_test_selected
    
    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using the specified method.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        self.logger.info("Scaling features...")
        
        method = self.config['scaling']['method']
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'none':
            self.logger.info("No scaling applied")
            return X_train, X_test
        else:
            self.logger.warning(f"Unknown scaling method: {method}")
            return X_train, X_test
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.scaler = scaler
        self.logger.info(f"Applied {method} scaling")
        
        return X_train_scaled, X_test_scaled
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores if available.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.selected_features:
            return pd.DataFrame()
        
        # Create a simple importance ranking based on selection order
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance_rank': range(len(self.selected_features))
        })
        
        return importance_df.sort_values('importance_rank')
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get a summary of the preprocessing steps applied.
        
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'scaling_method': self.config['scaling']['method'],
            'feature_selection_enabled': self.config['feature_selection']['enabled'],
            'outlier_detection_enabled': self.config['outlier_detection']['enabled'],
            'n_selected_features': len(self.selected_features),
            'is_fitted': self.is_fitted
        }
        
        if self.config['feature_selection']['enabled']:
            summary['feature_selection_method'] = self.config['feature_selection']['method']
        
        if self.config['outlier_detection']['enabled']:
            summary['outlier_detection_method'] = self.config['outlier_detection']['method']
        
        return summary
