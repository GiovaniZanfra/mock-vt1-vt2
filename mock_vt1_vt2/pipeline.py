"""
VT1/VT2 Prediction Pipeline

This module provides a comprehensive pipeline for predicting VT1 and VT2
from cycling data using feature engineering, data augmentation, and
machine learning models.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .feature_engineering import FeatureEngineer
from .data_augmentation import DataAugmenter
from .preprocessing import DataPreprocessor
from .model_trainer import ModelTrainer
from .evaluation import ModelEvaluator


class VTPredictionPipeline:
    """
    Main pipeline class for VT1/VT2 prediction.
    """
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config['feature_engineering'])
        self.data_augmenter = DataAugmenter(self.config['data_augmentation'])
        self.preprocessor = DataPreprocessor(self.config['preprocessing'])
        self.model_trainer = ModelTrainer(self.config['model'])
        self.evaluator = ModelEvaluator(self.config['evaluation'])
        
        # Data storage
        self.raw_data = None
        self.feature_engineered_data = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.predictions = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config['logging']
        log_dir = Path(log_config['log_file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config['log_file']),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from the specified path.
        
        Returns:
            Loaded DataFrame
        """
        self.logger.info("Loading raw data...")
        data_path = self.config['data']['raw_data_path']
        self.raw_data = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(self.raw_data)} samples with {len(self.raw_data.columns)} features")
        return self.raw_data
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Apply feature engineering to the raw data.
        
        Returns:
            Feature engineered DataFrame
        """
        self.logger.info("Starting feature engineering...")
        
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_data() first.")
        
        self.feature_engineered_data = self.feature_engineer.transform(self.raw_data)
        
        # Save feature engineered data
        output_path = self.config['data']['feature_engineered_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.feature_engineered_data.to_csv(output_path, index=False)
        self.logger.info(f"Feature engineered data saved to {output_path}")
        self.logger.info(f"Feature engineering created {len(self.feature_engineered_data.columns)} features")
        
        return self.feature_engineered_data
    
    def augment_data(self) -> pd.DataFrame:
        """
        Apply data augmentation using TSAug.
        
        Returns:
            Augmented DataFrame
        """
        if not self.config['data_augmentation']['enabled']:
            self.logger.info("Data augmentation disabled in config")
            return self.feature_engineered_data
        
        self.logger.info("Starting data augmentation...")
        
        if self.feature_engineered_data is None:
            raise ValueError("Feature engineered data not available. Call engineer_features() first.")
        
        augmented_data = self.data_augmenter.augment(self.feature_engineered_data)
        
        # Save augmented data
        output_path = self.config['data']['processed_data_path'] + "augmented_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        augmented_data.to_csv(output_path, index=False)
        self.logger.info(f"Augmented data saved to {output_path}")
        self.logger.info(f"Data augmentation created {len(augmented_data)} samples")
        
        return augmented_data
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (train_data, test_data)
        """
        self.logger.info("Splitting data into train and test sets...")
        
        targets = self.config['model']['targets']
        feature_cols = [col for col in data.columns if col not in targets + ['idx', 'sid']]
        
        X = data[feature_cols]
        y = data[targets]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Combine features and targets
        self.train_data = pd.concat([X_train, y_train], axis=1)
        self.test_data = pd.concat([X_test, y_test], axis=1)
        
        # Save train/test data
        train_path = self.config['data']['train_data_path']
        test_path = self.config['data']['test_data_path']
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        self.train_data.to_csv(train_path, index=False)
        self.test_data.to_csv(test_path, index=False)
        
        self.logger.info(f"Train set: {len(self.train_data)} samples")
        self.logger.info(f"Test set: {len(self.test_data)} samples")
        
        return self.train_data, self.test_data
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data (scaling, feature selection, outlier detection).
        
        Returns:
            Tuple of (processed_train_data, processed_test_data)
        """
        self.logger.info("Starting data preprocessing...")
        
        if self.train_data is None or self.test_data is None:
            raise ValueError("Train/test data not available. Call split_data() first.")
        
        processed_train, processed_test = self.preprocessor.fit_transform(
            self.train_data, self.test_data
        )
        
        self.logger.info("Data preprocessing completed")
        return processed_train, processed_test
    
    def train_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Train models for VT1 and VT2 prediction.
        
        Args:
            train_data: Training data
            test_data: Test data
            
        Returns:
            Dictionary of trained models
        """
        self.logger.info("Starting model training...")
        
        self.models = self.model_trainer.train_all_models(train_data, test_data)
        
        self.logger.info(f"Trained {len(self.models)} models")
        return self.models
    
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate the trained models.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("Starting model evaluation...")
        
        evaluation_results = self.evaluator.evaluate_all_models(
            self.models, test_data
        )
        
        # Save evaluation results
        if self.config['output']['save_evaluation_report']:
            output_path = self.config['output']['evaluation_report_path']
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert results to DataFrame and save
            results_df = pd.DataFrame(evaluation_results).T
            results_df.to_csv(output_path)
            self.logger.info(f"Evaluation results saved to {output_path}")
        
        return evaluation_results
    
    def run_pipeline(self) -> Dict:
        """
        Run the complete pipeline from data loading to model evaluation.
        
        Returns:
            Dictionary containing pipeline results
        """
        self.logger.info("Starting VT1/VT2 prediction pipeline...")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Engineer features
            self.engineer_features()
            
            # Step 3: Augment data (if enabled)
            data_for_training = self.augment_data()
            
            # Step 4: Split data
            train_data, test_data = self.split_data(data_for_training)
            
            # Step 5: Preprocess data
            processed_train, processed_test = self.preprocess_data()
            
            # Step 6: Train models
            models = self.train_models(processed_train, processed_test)
            
            # Step 7: Evaluate models
            evaluation_results = self.evaluate_models(processed_test)
            
            self.logger.info("Pipeline completed successfully!")
            
            return {
                'models': models,
                'evaluation_results': evaluation_results,
                'feature_engineered_data': self.feature_engineered_data,
                'train_data': self.train_data,
                'test_data': self.test_data
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            new_data: New data for prediction
            
        Returns:
            DataFrame with predictions
        """
        if not self.models:
            raise ValueError("No trained models available. Run the pipeline first.")
        
        self.logger.info("Making predictions on new data...")
        
        # Apply feature engineering
        engineered_data = self.feature_engineer.transform(new_data)
        
        # Apply preprocessing
        processed_data = self.preprocessor.transform(engineered_data)
        
        # Make predictions
        predictions = {}
        for target, model in self.models.items():
            predictions[f'{target}_pred'] = model.predict(processed_data)
        
        # Combine predictions
        predictions_df = pd.DataFrame(predictions)
        result = pd.concat([new_data, predictions_df], axis=1)
        
        # Save predictions
        if self.config['output']['save_predictions']:
            output_path = self.config['output']['predictions_path']
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.to_csv(output_path, index=False)
            self.logger.info(f"Predictions saved to {output_path}")
        
        return result


def main():
    """Main function to run the pipeline."""
    pipeline = VTPredictionPipeline()
    results = pipeline.run_pipeline()
    
    print("Pipeline completed!")
    print(f"Trained {len(results['models'])} models")
    print("Evaluation results:")
    for target, metrics in results['evaluation_results'].items():
        print(f"\n{target}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
