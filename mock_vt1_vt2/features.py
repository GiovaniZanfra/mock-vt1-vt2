"""
Feature engineering for time series data to predict vt1 and vt2.
Optimized for embedded systems (smartwatches, cellphones).
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Tuple, Optional
from .config import *

logger = logging.getLogger(__name__)

class EmbeddedTimeSeriesFeatureExtractor:
    """
    Lightweight feature extractor optimized for embedded systems.
    Focuses on computational efficiency and small memory footprint.
    """
    
    def __init__(self, use_fft: bool = False):
        self.use_fft = use_fft  # FFT is expensive, can be disabled
        
    def extract_basic_statistics(self, series: pd.Series) -> dict:
        """
        Extract only the most essential statistical features.
        Very fast and memory efficient.
        """
        features = {}
        
        # Core statistics (fastest to compute)
        features['mean'] = series.mean()
        features['std'] = series.std()
        features['min'] = series.min()
        features['max'] = series.max()
        features['range'] = features['max'] - features['min']
        
        # Simple trend (linear fit)
        if len(series) > 1:
            features['trend'] = np.polyfit(range(len(series)), series, 1)[0]
        else:
            features['trend'] = 0
        
        return features
    
    def extract_shape_features(self, series: pd.Series) -> dict:
        """
        Extract simple shape-based features.
        """
        features = {}
        
        # Zero crossings (simple pattern detection)
        zero_crossings = np.sum(np.diff(np.signbit(series - series.mean())))
        features['zero_crossings'] = zero_crossings
        
        # Peak detection (limited to avoid memory issues)
        try:
            peaks, _ = stats.find_peaks(series, distance=10)  # Minimum distance between peaks
            features['n_peaks'] = len(peaks)
            if len(peaks) > 0:
                features['peak_mean'] = series.iloc[peaks].mean()
            else:
                features['peak_mean'] = 0
        except:
            features['n_peaks'] = 0
            features['peak_mean'] = 0
        
        return features
    
    def extract_rolling_features(self, series: pd.Series, window_size: int = 50) -> dict:
        """
        Extract rolling features with smaller window for efficiency.
        """
        features = {}
        
        # Use smaller window for embedded systems
        if len(series) >= window_size:
            rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
            rolling_std = series.rolling(window=window_size, min_periods=1).std()
            
            features['rolling_mean_mean'] = rolling_mean.mean()
            features['rolling_std_mean'] = rolling_std.mean()
        else:
            features['rolling_mean_mean'] = series.mean()
            features['rolling_std_mean'] = series.std()
        
        return features
    
    def extract_lightweight_frequency_features(self, series: pd.Series) -> dict:
        """
        Extract minimal frequency features if FFT is enabled.
        """
        features = {}
        
        if not self.use_fft:
            return features
        
        try:
            # Use only first few FFT components
            series_centered = series - series.mean()
            fft_vals = fft(series_centered)
            fft_magnitude = np.abs(fft_vals)
            
            # Only take first 3 components (very lightweight)
            features['fft_magnitude_1'] = fft_magnitude[1] if len(fft_magnitude) > 1 else 0
            features['fft_magnitude_2'] = fft_magnitude[2] if len(fft_magnitude) > 2 else 0
            features['fft_magnitude_3'] = fft_magnitude[3] if len(fft_magnitude) > 3 else 0
            
        except Exception as e:
            logger.warning(f"FFT feature extraction failed: {e}")
            features['fft_magnitude_1'] = 0
            features['fft_magnitude_2'] = 0
            features['fft_magnitude_3'] = 0
        
        return features
    
    def extract_all_features(self, series: pd.Series) -> dict:
        """
        Extract all lightweight features optimized for embedded systems.
        """
        features = {}
        
        # Basic statistics (fastest)
        features.update(self.extract_basic_statistics(series))
        
        # Shape features (moderate cost)
        features.update(self.extract_shape_features(series))
        
        # Rolling features (moderate cost)
        features.update(self.extract_rolling_features(series))
        
        # Frequency features (most expensive, optional)
        features.update(self.extract_lightweight_frequency_features(series))
        
        return features

class TimeSeriesFeatureExtractor:
    """
    Original feature extractor (kept for comparison).
    """
    
    def __init__(self, window_size: int = 100, fft_components: int = 10):
        self.window_size = window_size
        self.fft_components = fft_components
        self.scaler = StandardScaler()
        
    def extract_statistical_features(self, series: pd.Series) -> dict:
        """
        Extract statistical features from a time series.
        """
        features = {}
        
        # Basic statistics
        features['mean'] = series.mean()
        features['std'] = series.std()
        features['min'] = series.min()
        features['max'] = series.max()
        features['median'] = series.median()
        features['q25'] = series.quantile(0.25)
        features['q75'] = series.quantile(0.75)
        features['iqr'] = features['q75'] - features['q25']
        
        # Higher order moments
        features['skewness'] = series.skew()
        features['kurtosis'] = series.kurtosis()
        
        # Range and variability
        features['range'] = features['max'] - features['min']
        features['cv'] = features['std'] / features['mean'] if features['mean'] != 0 else 0
        
        # Trend features
        features['trend'] = np.polyfit(range(len(series)), series, 1)[0]
        
        return features
    
    def extract_frequency_features(self, series: pd.Series) -> dict:
        """
        Extract frequency domain features using FFT.
        """
        features = {}
        
        # Remove mean for FFT
        series_centered = series - series.mean()
        
        # Compute FFT
        fft_vals = fft(series_centered)
        fft_magnitude = np.abs(fft_vals)
        
        # Get dominant frequencies
        freqs = np.fft.fftfreq(len(series))
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = fft_magnitude[:len(freqs)//2]
        
        # Top frequency components
        top_indices = np.argsort(positive_magnitude)[-self.fft_components:]
        
        for i, idx in enumerate(top_indices):
            features[f'fft_freq_{i}'] = positive_freqs[idx]
            features[f'fft_magnitude_{i}'] = positive_magnitude[idx]
        
        # Spectral features
        features['spectral_centroid'] = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
        features['spectral_bandwidth'] = np.sqrt(np.sum(((positive_freqs - features['spectral_centroid']) ** 2) * positive_magnitude) / np.sum(positive_magnitude))
        
        return features
    
    def extract_shape_features(self, series: pd.Series) -> dict:
        """
        Extract shape-based features from time series.
        """
        features = {}
        
        # Peak features
        peaks, _ = stats.find_peaks(series)
        valleys, _ = stats.find_peaks(-series)
        
        features['n_peaks'] = len(peaks)
        features['n_valleys'] = len(valleys)
        
        if len(peaks) > 0:
            features['peak_mean'] = series.iloc[peaks].mean()
            features['peak_std'] = series.iloc[peaks].std()
        else:
            features['peak_mean'] = 0
            features['peak_std'] = 0
            
        if len(valleys) > 0:
            features['valley_mean'] = series.iloc[valleys].mean()
            features['valley_std'] = series.iloc[valleys].std()
        else:
            features['valley_mean'] = 0
            features['valley_std'] = 0
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(series - series.mean())))
        features['zero_crossings'] = zero_crossings
        
        return features
    
    def extract_rolling_features(self, series: pd.Series) -> dict:
        """
        Extract rolling window features.
        """
        features = {}
        
        # Rolling statistics
        rolling_mean = series.rolling(window=self.window_size, min_periods=1).mean()
        rolling_std = series.rolling(window=self.window_size, min_periods=1).std()
        
        features['rolling_mean_mean'] = rolling_mean.mean()
        features['rolling_mean_std'] = rolling_mean.std()
        features['rolling_std_mean'] = rolling_std.mean()
        features['rolling_std_std'] = rolling_std.std()
        
        # Rolling trend
        rolling_trend = []
        for i in range(len(series) - self.window_size + 1):
            window = series.iloc[i:i+self.window_size]
            if len(window) > 1:
                trend = np.polyfit(range(len(window)), window, 1)[0]
                rolling_trend.append(trend)
        
        if rolling_trend:
            features['rolling_trend_mean'] = np.mean(rolling_trend)
            features['rolling_trend_std'] = np.std(rolling_trend)
        else:
            features['rolling_trend_mean'] = 0
            features['rolling_trend_std'] = 0
        
        return features
    
    def extract_all_features(self, series: pd.Series) -> dict:
        """
        Extract all features from a time series.
        """
        features = {}
        
        # Statistical features
        features.update(self.extract_statistical_features(series))
        
        # Frequency features
        features.update(self.extract_frequency_features(series))
        
        # Shape features
        features.update(self.extract_shape_features(series))
        
        # Rolling features
        features.update(self.extract_rolling_features(series))
        
        return features

def create_features_from_dataframe(df: pd.DataFrame, target_cols: Optional[List[str]] = None, 
                                 embedded: bool = True) -> pd.DataFrame:
    """
    Create features from a dataframe containing time series data.
    """
    logger.info("Creating features from dataframe...")
    
    # Remove target columns if specified
    if target_cols:
        feature_cols = [col for col in df.columns if col not in target_cols]
        X = df[feature_cols].copy()
        y = df[target_cols].copy() if target_cols else None
    else:
        X = df.copy()
        y = None
    
    # Choose feature extractor based on embedded flag
    if embedded:
        extractor = EmbeddedTimeSeriesFeatureExtractor(use_fft=False)  # Disable FFT for embedded
        logger.info("Using embedded-optimized feature extractor")
    else:
        extractor = TimeSeriesFeatureExtractor()
        logger.info("Using full feature extractor")
    
    # Extract features for each column
    all_features = []
    
    for col in X.columns:
        logger.info(f"Extracting features from column: {col}")
        features = extractor.extract_all_features(X[col])
        features['column_name'] = col
        all_features.append(features)
    
    # Create features dataframe
    features_df = pd.DataFrame(all_features)
    
    # Pivot to get one row per sample
    if len(features_df) > 0:
        # If we have multiple columns, we need to aggregate features
        # For now, let's take the mean across all columns for each feature
        feature_cols = [col for col in features_df.columns if col != 'column_name']
        aggregated_features = features_df[feature_cols].mean()
        
        # Create a single row with aggregated features
        final_features = pd.DataFrame([aggregated_features])
    else:
        final_features = pd.DataFrame()
    
    logger.info(f"Created features with shape: {final_features.shape}")
    
    return final_features

def save_features(features_df: pd.DataFrame, filename: str = "engineered_features.csv") -> None:
    """
    Save engineered features to data/interim/
    """
    features_path = DATA_INTERIM / filename
    features_df.to_csv(features_path, index=False)
    logger.info(f"Saved engineered features to {features_path}")

def main():
    """
    Main feature engineering pipeline
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Load interim data
    try:
        interim_data = pd.read_csv(INTERIM_DATA_FILE)
        logger.info(f"Loaded interim data with shape: {interim_data.shape}")
    except FileNotFoundError:
        logger.error("Interim data not found. Run dataset.py first.")
        return
    
    # Create features (embedded version)
    features_df = create_features_from_dataframe(interim_data, embedded=True)
    
    # Save features
    save_features(features_df)
    
    logger.info("Feature engineering pipeline completed!")

if __name__ == "__main__":
    main()
