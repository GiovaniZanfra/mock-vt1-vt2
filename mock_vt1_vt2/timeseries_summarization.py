"""
Timeseries Summarization Module.

This module processes raw timeseries data and converts it to interim summarized features.
It supports multiple algorithms including legacy C-process functions and new implementations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
import yaml

from .legacy import c_process_features_array


class TimeseriesSummarizer:
    """
    Timeseries summarization processor that supports multiple algorithms.
    
    This class handles the conversion of raw timeseries data to interim
    summarized features using configurable algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the summarizer with configuration.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
        self.algorithm = config.get("algorithm", "basic_moving_average")
        self.window_size = config.get("window_size", 30)
        self.output_dir = Path(config.get("output_dir", "data/interim"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Algorithm-specific parameters
        self.algorithm_params = config.get("algorithm_params", {})
        
        logger.info(f"Initialized TimeseriesSummarizer with algorithm: {self.algorithm}")
    
    def process_dataset(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the entire dataset using the configured algorithm.
        
        Args:
            raw_data: Raw dataset with timeseries data
            
        Returns:
            Processed dataset with summarized features
        """
        logger.info(f"Processing dataset with {len(raw_data)} rows using {self.algorithm}")
        
        results = []
        
        for idx, row in raw_data.iterrows():
            try:
                features = self._process_single_series(row)
                if features is not None:
                    features["idx"] = row.get("idx", idx)
                    results.append(features)
            except Exception as e:
                logger.warning(f"Failed to process series {idx}: {e}")
                continue
        
        if not results:
            raise ValueError("No series were successfully processed")
        
        processed_df = pd.DataFrame(results)
        logger.info(f"Successfully processed {len(processed_df)} series")
        
        return processed_df
    
    def _process_single_series(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Process a single timeseries row using the configured algorithm.
        
        Args:
            row: Single row containing timeseries data
            
        Returns:
            Dictionary of computed features or None if processing failed
        """
        # Extract timeseries data
        hr_series = self._extract_series(row, "hr")
        speed_series = self._extract_series(row, "speed")
        powermeter_series = self._extract_series(row, "powermeter")
        
        if hr_series is None or speed_series is None:
            return None
        
        # Route to appropriate algorithm
        if self.algorithm == "ftp_cycling":
            return self._process_ftp_cycling(hr_series, speed_series, powermeter_series)
        elif self.algorithm == "vo2max_cycling":
            return self._process_vo2max_cycling(hr_series, speed_series, powermeter_series)
        elif self.algorithm == "basic_moving_average":
            return self._process_basic_moving_average(hr_series, speed_series, powermeter_series)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _extract_series(self, row: pd.Series, column: str) -> Optional[pd.Series]:
        """Extract and validate timeseries data from row."""
        data = row.get(column)
        if data is None or pd.isna(data):
            return None
        
        if isinstance(data, list):
            series = pd.Series(data)
        elif isinstance(data, pd.Series):
            series = data
        else:
            logger.warning(f"Unexpected data type for {column}: {type(data)}")
            return None
        
        # Remove invalid values
        series = series.replace([np.inf, -np.inf], np.nan)
        series = series.dropna()
        
        if len(series) < self.window_size:
            logger.warning(f"Insufficient data for {column}: {len(series)} < {self.window_size}")
            return None
        
        return series
    
    def _process_ftp_cycling(self, hr: pd.Series, speed: pd.Series, 
                           powermeter: Optional[pd.Series]) -> Dict[str, Any]:
        """Process using FTP cycling algorithm from legacy code."""
        try:
            # Use legacy C-process function
            result = c_process_features_array(
                hr=hr,
                speed=speed,
                powermeter=powermeter if powermeter is not None else [],
                algorithm="ftp-cycling",
                verbose=self.algorithm_params.get("verbose", False)
            )
            
            return {
                "hr_mean": result[0],
                "hr_std": result[1],
                "speed_mean": result[2],
                "speed_std": result[3],
                "gain_mean": result[4],
                "gain_std": result[5],
                "percentile_hr": result[6],
                "percentile_gain": result[7],
                "algorithm": "ftp_cycling"
            }
        except Exception as e:
            logger.error(f"FTP cycling processing failed: {e}")
            raise
    
    def _process_vo2max_cycling(self, hr: pd.Series, speed: pd.Series,
                              powermeter: Optional[pd.Series]) -> Dict[str, Any]:
        """Process using VO2max cycling algorithm from legacy code."""
        try:
            # Use legacy C-process function
            result = c_process_features_array(
                hr=hr,
                speed=speed,
                powermeter=powermeter if powermeter is not None else [],
                algorithm="vo2max-cycling",
                verbose=self.algorithm_params.get("verbose", False)
            )
            
            return {
                "hr_mean": result[0],
                "hr_std": result[1],
                "speed_mean": result[2],
                "speed_std": result[3],
                "gain_mean": result[4],
                "gain_std": result[5],
                "percentile_hr": result[6],
                "percentile_gain": result[7],
                "algorithm": "vo2max_cycling"
            }
        except Exception as e:
            logger.error(f"VO2max cycling processing failed: {e}")
            raise
    
    def _process_basic_moving_average(self, hr: pd.Series, speed: pd.Series,
                                    powermeter: Optional[pd.Series]) -> Dict[str, Any]:
        """Process using basic moving average algorithm."""
        window = self.algorithm_params.get("window_size", self.window_size)
        
        # Compute moving averages
        hr_ma = hr.rolling(window=window, center=True).mean()
        speed_ma = speed.rolling(window=window, center=True).mean()
        
        # Compute moving medians
        hr_median = hr.rolling(window=window, center=True).median()
        speed_median = speed.rolling(window=window, center=True).median()
        
        # Compute gain (HR/Speed ratio)
        gain = hr / speed
        gain = gain.replace([np.inf, -np.inf], np.nan).dropna()
        gain_ma = gain.rolling(window=window, center=True).mean()
        
        # Extract features from the moving averages
        features = {
            "hr_mean": hr_ma.mean(),
            "hr_std": hr_ma.std(),
            "hr_median": hr_median.mean(),
            "hr_min": hr_ma.min(),
            "hr_max": hr_ma.max(),
            "speed_mean": speed_ma.mean(),
            "speed_std": speed_ma.std(),
            "speed_median": speed_median.mean(),
            "speed_min": speed_ma.min(),
            "speed_max": speed_ma.max(),
            "gain_mean": gain_ma.mean(),
            "gain_std": gain_ma.std(),
            "gain_min": gain_ma.min(),
            "gain_max": gain_ma.max(),
            "algorithm": "basic_moving_average"
        }
        
        # Add percentiles if requested
        if self.algorithm_params.get("include_percentiles", True):
            features.update({
                "hr_75th": hr_ma.quantile(0.75),
                "hr_90th": hr_ma.quantile(0.90),
                "speed_75th": speed_ma.quantile(0.75),
                "speed_90th": speed_ma.quantile(0.90),
                "gain_75th": gain_ma.quantile(0.75),
                "gain_90th": gain_ma.quantile(0.90),
            })
        
        return features
    
    def save_processed_data(self, processed_data: pd.DataFrame, 
                          filename: Optional[str] = None) -> Path:
        """
        Save processed data to interim directory.
        
        Args:
            processed_data: Processed dataset
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"timeseries_summary_{self.algorithm}_{timestamp}.parquet"
        
        output_path = self.output_dir / filename
        processed_data.to_parquet(output_path)
        
        logger.info(f"Saved processed data to: {output_path}")
        return output_path


def load_summarization_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load summarization configuration from file or environment.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.environ.get("SUMMARIZATION_CONFIG", "config/summarization.yaml")
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Summarization config not found: {config_file}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f) or {}
    
    logger.info(f"Loaded summarization config: {config_file}")
    return config


def process_raw_to_interim(raw_data_path: str, config_path: Optional[str] = None,
                          output_filename: Optional[str] = None) -> Path:
    """
    Process raw data to interim summarized features.
    
    Args:
        raw_data_path: Path to raw data file
        config_path: Path to summarization config
        output_filename: Optional output filename
        
    Returns:
        Path to saved interim data
    """
    # Load configuration
    config = load_summarization_config(config_path)
    
    # Load raw data
    raw_data = pd.read_parquet(raw_data_path)
    logger.info(f"Loaded raw data: {raw_data_path} with {len(raw_data)} rows")
    
    # Initialize summarizer
    summarizer = TimeseriesSummarizer(config)
    
    # Process data
    processed_data = summarizer.process_dataset(raw_data)
    
    # Save results
    output_path = summarizer.save_processed_data(processed_data, output_filename)
    
    return output_path


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process raw timeseries data to interim features")
    parser.add_argument(
        "--raw-data", 
        type=str, 
        required=True,
        help="Path to raw data file (Parquet format)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to summarization config file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output filename (optional)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = process_raw_to_interim(
            raw_data_path=args.raw_data,
            config_path=args.config,
            output_filename=args.output
        )
        print(f"✅ Processing completed successfully!")
        print(f"📁 Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
