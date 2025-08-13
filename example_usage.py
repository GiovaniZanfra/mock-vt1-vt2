#!/usr/bin/env python3
"""
Example Usage of VT1/VT2 Prediction Pipeline

This script demonstrates how to use the pipeline for different scenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mock_vt1_vt2.pipeline import VTPredictionPipeline
from mock_vt1_vt2.feature_engineering import FeatureEngineer
from mock_vt1_vt2.data_augmentation import DataAugmenter
from mock_vt1_vt2.preprocessing import DataPreprocessor
from mock_vt1_vt2.model_trainer import ModelTrainer
from mock_vt1_vt2.evaluation import ModelEvaluator


def example_1_basic_pipeline():
    """Example 1: Run the complete pipeline with default configuration."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Pipeline")
    print("=" * 60)
    
    # Initialize pipeline with default config
    pipeline = VTPredictionPipeline('config/pipeline_config.yaml')
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Print results
    print(f"\nTrained {len(results['models'])} models:")
    for target, model_info in results['models'].items():
        print(f"  - {target}: {model_info['algorithm']}")
    
    print("\nEvaluation Results:")
    for target, metrics in results['evaluation_results'].items():
        print(f"\n{target.upper()}:")
        for metric, value in metrics['metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    return results


def example_2_custom_feature_engineering():
    """Example 2: Custom feature engineering configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Feature Engineering")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/raw/cycling_train.csv')
    print(f"Loaded {len(data)} samples")
    
    # Create custom feature engineering config
    custom_config = {
        'demographics': {
            'include_bmi': True,
            'include_age_groups': True,
            'age_group_bins': [20, 30, 40, 50, 60, 70],
            'include_gender_encoding': True
        },
        'statistical_features': {
            'include_percentiles': True,
            'percentiles': [25, 50, 75],
            'include_skewness': True,
            'include_kurtosis': False,
            'include_entropy': False,
            'include_autocorrelation': False,
            'autocorrelation_lags': [1, 2, 3]
        },
        'ratio_features': {
            'include_hr_speed_ratio': True,
            'include_gain_speed_ratio': True,
            'include_hr_gain_ratio': False,
            'include_hr_variability': True
        },
        'interaction_features': {
            'include_age_weight_interaction': True,
            'include_age_height_interaction': False,
            'include_weight_height_interaction': False,
            'include_hr_speed_interaction': True
        },
        'polynomial_features': {
            'enabled': False,
            'degree': 2,
            'include_features': ['age', 'weight', 'height', 'hr_mean', 'speed_mean']
        },
        'domain_features': {
            'include_hr_reserve': True,
            'include_speed_efficiency': True,
            'include_gain_efficiency': False,
            'include_hr_response': True
        }
    }
    
    # Apply feature engineering
    engineer = FeatureEngineer(custom_config)
    engineered_data = engineer.transform(data)
    
    print(f"Feature engineering created {len(engineered_data.columns)} features")
    print(f"New features: {list(engineered_data.columns[-10:])}")  # Show last 10 features
    
    return engineered_data


def example_3_data_augmentation():
    """Example 3: Data augmentation with TSAug."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Data Augmentation")
    print("=" * 60)
    
    # Load feature engineered data
    data = pd.read_csv('data/processed/feature_engineered_data.csv')
    print(f"Original data: {len(data)} samples")
    
    # Create custom augmentation config
    augmentation_config = {
        'enabled': True,
        'augmentation_factor': 2,  # Create 2x more samples
        'tsaug': {
            'time_warping': {
                'enabled': True,
                'probability': 0.3,
                'strength': 0.05
            },
            'magnitude_warping': {
                'enabled': True,
                'probability': 0.3,
                'strength': 0.05
            },
            'jittering': {
                'enabled': True,
                'probability': 0.2,
                'strength': 0.02
            },
            'scaling': {
                'enabled': True,
                'probability': 0.2,
                'strength': 0.05
            },
            'rotation': {
                'enabled': False,
                'probability': 0.1,
                'strength': 0.02
            },
            'permutation': {
                'enabled': False,
                'probability': 0.05,
                'segment_size': 0.1
            }
        }
    }
    
    # Apply data augmentation
    augmenter = DataAugmenter(augmentation_config)
    augmented_data = augmenter.augment(data)
    
    print(f"Augmented data: {len(augmented_data)} samples")
    print(f"Created {len(augmented_data) - len(data)} synthetic samples")
    
    return augmented_data


def example_4_model_comparison():
    """Example 4: Compare different models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Model Comparison")
    print("=" * 60)
    
    # Load processed data
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Create model config with specific algorithms
    model_config = {
        'targets': ['vt1', 'vt2'],
        'model_type': 'regression',
        'algorithms': ['random_forest', 'xgboost', 'lightgbm'],
        'cv': {
            'method': 'kfold',
            'n_splits': 5,
            'random_state': 42
        },
        'hyperparameter_tuning': {
            'enabled': True,
            'method': 'random_search',
            'n_iterations': 20  # Reduced for faster execution
        }
    }
    
    # Train models
    trainer = ModelTrainer(model_config)
    models = trainer.train_all_models(train_data, test_data)
    
    # Get model summary
    summary = trainer.get_model_summary()
    print("\nModel Performance Summary:")
    print(summary.to_string(index=False))
    
    return models, summary


def example_5_prediction_on_new_data():
    """Example 5: Make predictions on new data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Predictions on New Data")
    print("=" * 60)
    
    # Create sample new data
    new_data = pd.DataFrame({
        'idx': [1001, 1002, 1003],
        'sid': ['new_001', 'new_002', 'new_003'],
        'gender': ['M', 'F', 'M'],
        'age': [25, 35, 45],
        'weight': [70.0, 65.0, 80.0],
        'height': [175, 165, 180],
        'hr_rest': [60, 65, 70],
        'hr_mean': [120, 115, 125],
        'hr_std': [8, 6, 10],
        'speed_mean': [20.0, 18.0, 22.0],
        'speed_std': [2.5, 2.0, 3.0],
        'gain_mean': [150.0, 120.0, 180.0],
        'gain_std': [30.0, 25.0, 35.0]
    })
    
    print("New data for prediction:")
    print(new_data[['age', 'weight', 'height', 'hr_mean', 'speed_mean']].to_string(index=False))
    
    # Initialize pipeline (assuming models are already trained)
    pipeline = VTPredictionPipeline('config/pipeline_config.yaml')
    
    # Load trained models (if available)
    try:
        pipeline.model_trainer.load_models('models/')
        print("\nLoaded pre-trained models")
        
        # Make predictions
        predictions = pipeline.predict(new_data)
        
        print("\nPredictions:")
        print(predictions[['idx', 'sid', 'vt1_pred', 'vt2_pred']].to_string(index=False))
        
    except FileNotFoundError:
        print("\nNo pre-trained models found. Please run the pipeline first.")
    
    return new_data


def main():
    """Run all examples."""
    print("VT1/VT2 Prediction Pipeline - Examples")
    print("=" * 60)
    
    try:
        # Example 1: Basic pipeline
        results_1 = example_1_basic_pipeline()
        
        # Example 2: Custom feature engineering
        engineered_data = example_2_custom_feature_engineering()
        
        # Example 3: Data augmentation
        augmented_data = example_3_data_augmentation()
        
        # Example 4: Model comparison
        models, summary = example_4_model_comparison()
        
        # Example 5: Predictions
        new_data = example_5_prediction_on_new_data()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
