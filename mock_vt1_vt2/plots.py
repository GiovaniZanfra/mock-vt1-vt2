"""
Visualization functions for the vt1-vt2 prediction project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .config import *

logger = logging.getLogger(__name__)

def setup_plotting_style():
    """
    Set up consistent plotting style.
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_data_overview(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Create overview plots of the raw data.
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: First few columns time series
    if len(df.columns) > 0:
        sample_cols = df.columns[:min(5, len(df.columns))]
        for col in sample_cols:
            axes[0, 0].plot(df[col].values[:1000], label=col, alpha=0.7)
        axes[0, 0].set_title('Sample Time Series (First 1000 points)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Distribution of first few columns
    if len(df.columns) > 0:
        sample_cols = df.columns[:min(3, len(df.columns))]
        for col in sample_cols:
            axes[0, 1].hist(df[col].values, bins=50, alpha=0.7, label=col)
        axes[0, 1].set_title('Distribution of Sample Columns')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation matrix (if reasonable number of columns)
    if len(df.columns) <= 20:
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   ax=axes[1, 0], cbar_kws={'shrink': 0.8})
        axes[1, 0].set_title('Correlation Matrix')
    else:
        # For too many columns, show correlation with first few
        sample_cols = df.columns[:10]
        corr_matrix = df[sample_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 0], cbar_kws={'shrink': 0.8})
        axes[1, 0].set_title('Correlation Matrix (First 10 columns)')
    
    # Plot 4: Missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        axes[1, 1].bar(range(len(missing_data)), missing_data.values)
        axes[1, 1].set_title('Missing Values per Column')
        axes[1, 1].set_xlabel('Column Index')
        axes[1, 1].set_ylabel('Missing Count')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Missing Values', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=14, fontweight='bold')
        axes[1, 1].set_title('Missing Values')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved data overview plot to {save_path}")
    
    plt.show()

def plot_feature_importance(model, feature_names: list, target_name: str, 
                          save_path: Optional[Path] = None):
    """
    Plot feature importance for a trained model.
    """
    setup_plotting_style()
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        logger.warning("Model doesn't have feature importance attribute")
        return
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    top_features = feature_importance_df.head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Feature Importance for {target_name}', fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.show()

def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                             target_name: str, save_path: Optional[Path] = None):
    """
    Plot predictions vs actual values.
    """
    setup_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Model Performance for {target_name}', fontsize=16, fontweight='bold')
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions vs Actual')
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved predictions plot to {save_path}")
    
    plt.show()

def plot_time_series_features(features_df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Plot distributions of engineered features.
    """
    setup_plotting_style()
    
    # Select numerical features
    numerical_features = features_df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_features) == 0:
        logger.warning("No numerical features to plot")
        return
    
    # Create subplots
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Distribution of Engineered Features', fontsize=16, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(numerical_features):
        row = i // n_cols
        col = i % n_cols
        
        axes[row, col].hist(features_df[feature].values, bins=30, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved features plot to {save_path}")
    
    plt.show()

def main():
    """
    Main plotting pipeline.
    """
    logger.info("Starting plotting pipeline...")
    
    # Create reports directory
    reports_figures_dir = REPORTS_DIR / "figures"
    reports_figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load raw data and create overview plot
        raw_data = pd.read_csv(RAW_DATA_FILE)
        plot_data_overview(raw_data, save_path=reports_figures_dir / "data_overview.png")
        
        # Load interim data and create overview plot
        interim_data = pd.read_csv(INTERIM_DATA_FILE)
        plot_data_overview(interim_data, save_path=reports_figures_dir / "interim_data_overview.png")
        
        # Load features and create feature distribution plot
        features_path = DATA_INTERIM / "engineered_features.csv"
        if features_path.exists():
            features_df = pd.read_csv(features_path)
            plot_time_series_features(features_df, save_path=reports_figures_dir / "features_distribution.png")
        
        logger.info("Plotting pipeline completed!")
        
    except Exception as e:
        logger.error(f"Error during plotting: {e}")

if __name__ == "__main__":
    main()
