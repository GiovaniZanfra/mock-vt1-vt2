"""
Data Augmentation Module for VT1/VT2 Prediction

This module handles data augmentation using TSAug for time series data
and various oversampling techniques for imbalanced datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.utils import resample
import warnings

# Try to import imbalanced-learn, but handle case where it's not installed
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Try to import TSAug, but handle case where it's not installed
try:
    import tsaug
    TSAUG_AVAILABLE = True
except ImportError:
    TSAUG_AVAILABLE = False
    warnings.warn("TSAug not available. Install with: pip install tsaug")


class DataAugmenter:
    """
    Data augmentation class for VT1/VT2 prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data augmenter with configuration.
        
        Args:
            config: Data augmentation configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not TSAUG_AVAILABLE and config['enabled']:
            self.logger.warning("TSAug not available. Data augmentation will be limited.")
    
    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data augmentation to the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Augmented DataFrame
        """
        if not self.config['enabled']:
            return data
        
        self.logger.info("Starting data augmentation...")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Apply TSAug augmentation
        if TSAUG_AVAILABLE:
            df = self._apply_tsaug_augmentation(df)
        
        # Apply oversampling if enabled
        if self.config.get('oversampling', {}).get('enabled', False):
            df = self._apply_oversampling(df)
        
        self.logger.info(f"Data augmentation completed. Final shape: {df.shape}")
        return df
    
    def _apply_tsaug_augmentation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply TSAug augmentation to time series features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Augmented DataFrame
        """
        if not TSAUG_AVAILABLE:
            return data
        
        self.logger.info("Applying TSAug augmentation...")
        
        # Identify time series features (features with mean/std)
        ts_features = ['hr_mean', 'hr_std', 'speed_mean', 'speed_std', 'gain_mean', 'gain_std']
        available_ts_features = [f for f in ts_features if f in data.columns]
        
        if not available_ts_features:
            self.logger.warning("No time series features found for augmentation")
            return data
        
        # Create augmented samples
        augmentation_factor = self.config['augmentation_factor']
        augmented_samples = []
        
        for _ in range(augmentation_factor):
            # Create synthetic samples by applying TSAug transformations
            synthetic_sample = self._create_synthetic_sample(data, available_ts_features)
            augmented_samples.append(synthetic_sample)
        
        # Combine original and augmented data
        augmented_df = pd.concat([data] + augmented_samples, ignore_index=True)
        
        # Ensure all column names are strings
        augmented_df.columns = augmented_df.columns.astype(str)
        
        self.logger.info(f"TSAug created {len(augmented_samples)} synthetic samples")
        return augmented_df
    
    def _create_synthetic_sample(self, data: pd.DataFrame, ts_features: List[str]) -> pd.DataFrame:
        """
        Create a synthetic sample using TSAug transformations.
        
        Args:
            data: Original data
            ts_features: List of time series features
            
        Returns:
            Synthetic sample DataFrame
        """
        # Randomly select a base sample
        base_idx = np.random.randint(0, len(data))
        synthetic_sample = data.iloc[base_idx].copy()
        
        # Apply TSAug transformations to time series features
        tsaug_config = self.config['tsaug']
        
        for feature in ts_features:
            original_value = synthetic_sample[feature]
            
            # Apply transformations based on configuration
            transformed_value = original_value
            
            # Time warping (simulated for scalar values)
            if tsaug_config['time_warping']['enabled'] and np.random.random() < tsaug_config['time_warping']['probability']:
                strength = tsaug_config['time_warping']['strength']
                warp_factor = 1 + np.random.normal(0, strength)
                transformed_value *= warp_factor
            
            # Magnitude warping
            if tsaug_config['magnitude_warping']['enabled'] and np.random.random() < tsaug_config['magnitude_warping']['probability']:
                strength = tsaug_config['magnitude_warping']['strength']
                warp_factor = 1 + np.random.normal(0, strength)
                transformed_value *= warp_factor
            
            # Jittering
            if tsaug_config['jittering']['enabled'] and np.random.random() < tsaug_config['jittering']['probability']:
                strength = tsaug_config['jittering']['strength']
                jitter = np.random.normal(0, strength * original_value)
                transformed_value += jitter
            
            # Scaling
            if tsaug_config['scaling']['enabled'] and np.random.random() < tsaug_config['scaling']['probability']:
                strength = tsaug_config['scaling']['strength']
                scale_factor = 1 + np.random.normal(0, strength)
                transformed_value *= scale_factor
            
            # Ensure the transformed value is reasonable
            transformed_value = max(0, transformed_value)  # Non-negative for most features
            
            synthetic_sample[feature] = transformed_value
        
        # Add some noise to other numerical features
        numerical_features = data.select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            if feature not in ts_features and feature not in ['idx', 'sid', 'vt1', 'vt2']:
                original_value = synthetic_sample[feature]
                # Add small random noise
                noise = np.random.normal(0, 0.01 * abs(original_value))
                synthetic_sample[feature] = original_value + noise
        
        return synthetic_sample
    
    def _apply_oversampling(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply oversampling techniques for imbalanced datasets.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Oversampled DataFrame
        """
        self.logger.info("Applying oversampling...")
        
        oversampling_config = self.config.get('oversampling', {})
        method = oversampling_config.get('method', 'smote')
        
        # Separate features and targets
        target_cols = ['vt1', 'vt2']
        feature_cols = [col for col in data.columns if col not in target_cols + ['idx', 'sid']]
        
        X = data[feature_cols]
        y = data[target_cols]
        
        # Apply oversampling
        if method == 'smote':
            if not IMBALANCED_LEARN_AVAILABLE:
                self.logger.warning("SMOTE not available. Using random oversampling instead.")
                X_resampled, y_resampled = self._random_oversample(X, y)
            else:
                smote = SMOTE(
                    k_neighbors=oversampling_config.get('k_neighbors', 5),
                    random_state=oversampling_config.get('random_state', 42)
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
            
        elif method == 'adasyn':
            if not IMBALANCED_LEARN_AVAILABLE:
                self.logger.warning("ADASYN not available. Using random oversampling instead.")
                X_resampled, y_resampled = self._random_oversample(X, y)
            else:
                adasyn = ADASYN(
                    k_neighbors=oversampling_config.get('k_neighbors', 5),
                    random_state=oversampling_config.get('random_state', 42)
                )
                X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
        elif method == 'random_oversampling':
            # Simple random oversampling
            X_resampled, y_resampled = self._random_oversample(X, y)
            
        else:
            self.logger.warning(f"Unknown oversampling method: {method}")
            return data
        
        # Combine features and targets
        resampled_data = pd.concat([
            pd.DataFrame(X_resampled, columns=feature_cols),
            pd.DataFrame(y_resampled, columns=target_cols)
        ], axis=1)
        
        # Add synthetic indices and subject IDs
        resampled_data['idx'] = range(len(data), len(data) + len(resampled_data))
        resampled_data['sid'] = [f'synthetic_{i}' for i in range(len(resampled_data))]
        
        # Ensure all column names are strings
        resampled_data.columns = resampled_data.columns.astype(str)
        
        self.logger.info(f"Oversampling created {len(resampled_data)} samples")
        return resampled_data
    
    def _random_oversample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple:
        """
        Apply random oversampling.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        # Combine X and y for resampling
        combined = pd.concat([X, y], axis=1)
        
        # Determine target class for oversampling (use VT1 as reference)
        target_counts = y['vt1'].value_counts()
        majority_class = target_counts.idxmax()
        minority_class = target_counts.idxmin()
        
        # Separate majority and minority classes
        majority_samples = combined[y['vt1'] == majority_class]
        minority_samples = combined[y['vt1'] == minority_class]
        
        # Upsample minority class
        minority_upsampled = resample(
            minority_samples,
            replace=True,
            n_samples=len(majority_samples),
            random_state=42
        )
        
        # Combine majority and upsampled minority
        resampled = pd.concat([majority_samples, minority_upsampled])
        
        # Separate features and targets
        feature_cols = X.columns
        target_cols = y.columns
        
        X_resampled = resampled[feature_cols]
        y_resampled = resampled[target_cols]
        
        return X_resampled, y_resampled
    
    def _validate_augmented_data(self, original_data: pd.DataFrame, augmented_data: pd.DataFrame) -> bool:
        """
        Validate augmented data quality.
        
        Args:
            original_data: Original DataFrame
            augmented_data: Augmented DataFrame
            
        Returns:
            True if validation passes, False otherwise
        """
        # Check for reasonable value ranges
        numerical_cols = original_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in ['idx', 'sid']:
                continue
                
            original_range = (original_data[col].min(), original_data[col].max())
            augmented_range = (augmented_data[col].min(), augmented_data[col].max())
            
            # Check if augmented range is reasonable (within 50% of original range)
            range_expansion = 0.5
            expected_min = original_range[0] - range_expansion * (original_range[1] - original_range[0])
            expected_max = original_range[1] + range_expansion * (original_range[1] - original_range[0])
            
            if augmented_range[0] < expected_min or augmented_range[1] > expected_max:
                self.logger.warning(f"Augmented data for {col} has values outside expected range")
                return False
        
        return True
