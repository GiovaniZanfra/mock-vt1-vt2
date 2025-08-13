"""
Model Evaluation Module for VT1/VT2 Prediction

This module handles model evaluation, feature importance analysis,
and residual analysis for VT1 and VT2 prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings

# Try to import matplotlib and seaborn, but handle case where they're not installed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Model evaluation class for VT1/VT2 prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model evaluator with configuration.
        
        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
        
    def evaluate_all_models(self, models: Dict, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate all trained models.
        
        Args:
            models: Dictionary of trained models
            test_data: Test DataFrame
            
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("Starting model evaluation...")
        
        # Separate features and targets
        target_cols = ['vt1', 'vt2']
        feature_cols = [col for col in test_data.columns if col not in target_cols + ['idx', 'sid']]
        
        X_test = test_data[feature_cols]
        y_test = test_data[target_cols]
        
        evaluation_results = {}
        
        for target, model_info in models.items():
            self.logger.info(f"Evaluating {target} model...")
            
            # Get model and predictions
            model = model_info['model']
            y_true = y_test[target]
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred)
            
            # Feature importance analysis
            feature_importance = None
            if self.config['feature_importance']['enabled']:
                feature_importance = self._analyze_feature_importance(
                    model, X_test, y_true, target
                )
            
            # Residual analysis
            residuals = None
            if self.config['residual_analysis']['enabled']:
                residuals = self._analyze_residuals(y_true, y_pred, target)
            
            evaluation_results[target] = {
                'metrics': metrics,
                'feature_importance': feature_importance,
                'residuals': residuals,
                'predictions': y_pred,
                'actual': y_true
            }
        
        self.evaluation_results = evaluation_results
        self.logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for metric in self.config['metrics']:
            if metric == 'mae':
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
            elif metric == 'mse':
                metrics['mse'] = mean_squared_error(y_true, y_pred)
            elif metric == 'rmse':
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == 'r2':
                metrics['r2'] = r2_score(y_true, y_pred)
            elif metric == 'mape':
                metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return metrics
    
    def _analyze_feature_importance(self, model, X_test: pd.DataFrame, y_true: pd.Series, target: str) -> pd.DataFrame:
        """
        Analyze feature importance using permutation importance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_true: True values
            target: Target variable name
            
        Returns:
            DataFrame with feature importance scores
        """
        self.logger.info(f"Analyzing feature importance for {target}...")
        
        method = self.config['feature_importance']['method']
        
        if method == 'permutation':
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_true, 
                n_repeats=10, 
                random_state=42,
                scoring='neg_mean_absolute_error'
            )
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': X_test.columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance_mean', ascending=False)
            
        elif method == 'shap':
            # SHAP analysis (if available)
            try:
                import shap
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                # Calculate mean absolute SHAP values
                mean_shap = np.mean(np.abs(shap_values), axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance_mean': mean_shap,
                    'importance_std': np.std(np.abs(shap_values), axis=0)
                })
                
                importance_df = importance_df.sort_values('importance_mean', ascending=False)
                
            except ImportError:
                self.logger.warning("SHAP not available. Using permutation importance instead.")
                return self._analyze_feature_importance(model, X_test, y_true, target)
        
        else:
            self.logger.warning(f"Unknown feature importance method: {method}")
            return pd.DataFrame()
        
        return importance_df
    
    def _analyze_residuals(self, y_true: pd.Series, y_pred: np.ndarray, target: str) -> Dict:
        """
        Analyze model residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target: Target variable name
            
        Returns:
            Dictionary with residual analysis results
        """
        self.logger.info(f"Analyzing residuals for {target}...")
        
        residuals = y_true - y_pred
        
        # Calculate residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': self._calculate_skewness(residuals),
            'kurtosis': self._calculate_kurtosis(residuals),
            'normality_test': self._test_normality(residuals)
        }
        
        return {
            'residuals': residuals,
            'statistics': residual_stats
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        from scipy.stats import skew
        return skew(data)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        from scipy.stats import kurtosis
        return kurtosis(data)
    
    def _test_normality(self, data: np.ndarray) -> Dict:
        """Test normality of data using Shapiro-Wilk test."""
        from scipy.stats import shapiro
        
        try:
            statistic, p_value = shapiro(data)
            return {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except:
            return {
                'statistic': None,
                'p_value': None,
                'is_normal': None
            }
    
    def create_evaluation_plots(self, output_path: str = "reports/plots/"):
        """
        Create evaluation plots and save them to disk.
        
        Args:
            output_path: Path to save plots
        """
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib/Seaborn not available. Skipping plot generation.")
            return
            
        if not self.evaluation_results:
            self.logger.warning("No evaluation results available for plotting")
            return
        
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        for target, results in self.evaluation_results.items():
            self._create_target_plots(target, results, output_path)
        
        self.logger.info(f"Evaluation plots saved to {output_path}")
    
    def _create_target_plots(self, target: str, results: Dict, output_path: str):
        """Create plots for a specific target."""
        y_true = results['actual']
        y_pred = results['predictions']
        residuals = results['residuals']['residuals'] if results['residuals'] else None
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation Results - {target.upper()}', fontsize=16)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = results['metrics']['r2']
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Residuals vs Predicted
        if residuals is not None:
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals vs Predicted')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals histogram
        if residuals is not None:
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residuals Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance
        if results['feature_importance'] is not None:
            importance_df = results['feature_importance'].head(10)
            axes[1, 1].barh(range(len(importance_df)), importance_df['importance_mean'])
            axes[1, 1].set_yticks(range(len(importance_df)))
            axes[1, 1].set_yticklabels(importance_df['feature'])
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/{target}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, output_path: str = "reports/evaluation_report.txt"):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        if not self.evaluation_results:
            self.logger.warning("No evaluation results available for report")
            return
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("VT1/VT2 Prediction Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for target, results in self.evaluation_results.items():
                f.write(f"{target.upper()} Model Results\n")
                f.write("-" * 30 + "\n")
                
                # Metrics
                f.write("Performance Metrics:\n")
                for metric, value in results['metrics'].items():
                    f.write(f"  {metric.upper()}: {value:.4f}\n")
                
                # Feature importance
                if results['feature_importance'] is not None:
                    f.write("\nTop 10 Most Important Features:\n")
                    top_features = results['feature_importance'].head(10)
                    for idx, row in top_features.iterrows():
                        f.write(f"  {row['feature']}: {row['importance_mean']:.4f}\n")
                
                # Residual analysis
                if results['residuals']:
                    f.write("\nResidual Analysis:\n")
                    stats = results['residuals']['statistics']
                    f.write(f"  Mean: {stats['mean']:.4f}\n")
                    f.write(f"  Std: {stats['std']:.4f}\n")
                    f.write(f"  Skewness: {stats['skewness']:.4f}\n")
                    f.write(f"  Kurtosis: {stats['kurtosis']:.4f}\n")
                    if stats['normality_test']['is_normal'] is not None:
                        f.write(f"  Normal distribution: {stats['normality_test']['is_normal']}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        self.logger.info(f"Evaluation report saved to {output_path}")
    
    def get_best_model_performance(self) -> Dict:
        """
        Get the best performing model for each target.
        
        Returns:
            Dictionary with best model performance
        """
        best_performance = {}
        
        for target, results in self.evaluation_results.items():
            metrics = results['metrics']
            best_performance[target] = {
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'mape': metrics['mape']
            }
        
        return best_performance
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance across different models.
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for target, results in self.evaluation_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'target': target,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'mape': metrics['mape']
            })
        
        return pd.DataFrame(comparison_data)
