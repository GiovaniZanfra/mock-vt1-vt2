"""
Feature Engineering Module for VT1/VT2 Prediction

This module handles all feature engineering tasks including:
- Demographic features (BMI, age groups, gender encoding)
- Statistical features from time series data
- Ratio features
- Interaction features
- Polynomial features
- Domain-specific features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from scipy.stats import entropy
import logging


class FeatureEngineer:
    """
    Feature engineering class for VT1/VT2 prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Feature engineering configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.poly_transformer = None
        self.feature_names = []
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Feature engineered DataFrame
        """
        self.logger.info("Starting feature engineering...")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Apply demographic features
        if self.config['demographics']:
            df = self._add_demographic_features(df)
        
        # Apply statistical features
        if self.config['statistical_features']:
            df = self._add_statistical_features(df)
        
        # Apply ratio features
        if self.config['ratio_features']:
            df = self._add_ratio_features(df)
        
        # Apply interaction features
        if self.config['interaction_features']:
            df = self._add_interaction_features(df)
        
        # Apply polynomial features
        if self.config['polynomial_features']['enabled']:
            df = self._add_polynomial_features(df)
        
        # Apply domain-specific features
        if self.config['domain_features']:
            df = self._add_domain_features(df)
        
        # Remove any infinite or NaN values
        df = self._clean_features(df)
        
        # Ensure all column names are strings
        df.columns = df.columns.astype(str)
        
        self.logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def _add_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add demographic features."""
        self.logger.info("Adding demographic features...")
        
        config = self.config['demographics']
        
        # BMI calculation
        if config['include_bmi']:
            df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        
        # Age groups
        if config['include_age_groups']:
            age_bins = config['age_group_bins']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=False, include_lowest=True)
            # One-hot encode age groups
            age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
            df = pd.concat([df, age_dummies], axis=1)
            df.drop('age_group', axis=1, inplace=True)
        
        # Gender encoding
        if config['include_gender_encoding']:
            df['gender_encoded'] = (df['gender'] == 'M').astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features from time series data."""
        self.logger.info("Adding statistical features...")
        
        config = self.config['statistical_features']
        
        # Features to calculate statistics for
        ts_features = ['hr_mean', 'hr_std', 'speed_mean', 'speed_std', 'gain_mean', 'gain_std']
        
        for feature in ts_features:
            if feature in df.columns:
                # Percentiles
                if config['include_percentiles']:
                    percentiles = config['percentiles']
                    for p in percentiles:
                        df[f'{feature}_p{p}'] = np.percentile(df[feature], p)
                
                # Skewness
                if config['include_skewness']:
                    df[f'{feature}_skewness'] = stats.skew(df[feature])
                
                # Kurtosis
                if config['include_kurtosis']:
                    df[f'{feature}_kurtosis'] = stats.kurtosis(df[feature])
                
                # Entropy (discretized)
                if config['include_entropy']:
                    # Discretize for entropy calculation
                    bins = np.histogram_bin_edges(df[feature], bins='auto')
                    digitized = np.digitize(df[feature], bins)
                    df[f'{feature}_entropy'] = entropy(np.bincount(digitized))
                
                # Autocorrelation (simulated for individual features)
                if config['include_autocorrelation']:
                    lags = config['autocorrelation_lags']
                    for lag in lags:
                        if len(df) > lag:
                            # Calculate autocorrelation for the feature
                            autocorr = np.corrcoef(df[feature][:-lag], df[feature][lag:])[0, 1]
                            df[f'{feature}_autocorr_lag{lag}'] = autocorr if not np.isnan(autocorr) else 0
                        else:
                            df[f'{feature}_autocorr_lag{lag}'] = 0
        
        return df
    
    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ratio features."""
        self.logger.info("Adding ratio features...")
        
        config = self.config['ratio_features']
        
        # Heart rate to speed ratio
        if config['include_hr_speed_ratio']:
            df['hr_speed_ratio'] = df['hr_mean'] / (df['speed_mean'] + 1e-8)
        
        # Gain to speed ratio
        if config['include_gain_speed_ratio']:
            df['gain_speed_ratio'] = df['gain_mean'] / (df['speed_mean'] + 1e-8)
        
        # Heart rate to gain ratio
        if config['include_hr_gain_ratio']:
            df['hr_gain_ratio'] = df['hr_mean'] / (df['gain_mean'] + 1e-8)
        
        # Heart rate variability
        if config['include_hr_variability']:
            df['hr_variability'] = df['hr_std'] / (df['hr_mean'] + 1e-8)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features."""
        self.logger.info("Adding interaction features...")
        
        config = self.config['interaction_features']
        
        # Age-weight interaction
        if config['include_age_weight_interaction']:
            df['age_weight_interaction'] = df['age'] * df['weight']
        
        # Age-height interaction
        if config['include_age_height_interaction']:
            df['age_height_interaction'] = df['age'] * df['height']
        
        # Weight-height interaction
        if config['include_weight_height_interaction']:
            df['weight_height_interaction'] = df['weight'] * df['height']
        
        # Heart rate-speed interaction
        if config['include_hr_speed_interaction']:
            df['hr_speed_interaction'] = df['hr_mean'] * df['speed_mean']
        
        return df
    
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features."""
        self.logger.info("Adding polynomial features...")
        
        config = self.config['polynomial_features']
        degree = config['degree']
        include_features = config['include_features']
        
        # Select features for polynomial transformation
        poly_features = [f for f in include_features if f in df.columns]
        
        if poly_features:
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_data = poly.fit_transform(df[poly_features])
            
            # Create feature names
            feature_names = poly.get_feature_names_out(poly_features)
            
            # Add polynomial features to dataframe
            poly_df = pd.DataFrame(poly_data, columns=feature_names, index=df.index)
            
            # Remove original features to avoid duplication
            poly_df = poly_df.drop(columns=poly_features)
            
            # Add to main dataframe
            df = pd.concat([df, poly_df], axis=1)
            
            self.poly_transformer = poly
        
        return df
    
    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific features."""
        self.logger.info("Adding domain-specific features...")
        
        config = self.config['domain_features']
        
        # Heart rate reserve
        if config['include_hr_reserve']:
            df['hr_reserve'] = df['hr_mean'] - df['hr_rest']
        
        # Speed efficiency
        if config['include_speed_efficiency']:
            df['speed_efficiency'] = df['speed_mean'] / (df['hr_mean'] + 1e-8)
        
        # Gain efficiency
        if config['include_gain_efficiency']:
            df['gain_efficiency'] = df['gain_mean'] / (df['speed_mean'] + 1e-8)
        
        # Heart rate response
        if config['include_hr_response']:
            df['hr_response'] = (df['hr_mean'] - df['hr_rest']) / (df['speed_mean'] + 1e-8)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by removing infinite and NaN values."""
        self.logger.info("Cleaning features...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def get_feature_importance_ranking(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance ranking from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance ranking
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            return importance_df.sort_values('importance', ascending=False)
        else:
            self.logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
