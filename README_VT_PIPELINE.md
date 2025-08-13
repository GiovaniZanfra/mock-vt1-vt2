# VT1/VT2 Prediction Pipeline

A comprehensive machine learning pipeline for predicting VT1 and VT2 (ventilatory thresholds) from cycling data using feature engineering, data augmentation, and multiple machine learning algorithms.

## Overview

This pipeline is designed to predict VT1 and VT2 values from cycling performance data, including:
- **Demographics**: Age, gender, weight, height
- **Time series summaries**: Heart rate and speed statistics (mean, std)
- **Performance metrics**: Gain statistics and other derived features

## Features

### 🚀 **Comprehensive Feature Engineering**
- **Demographic features**: BMI, age groups, gender encoding
- **Statistical features**: Percentiles, skewness, kurtosis, entropy, autocorrelation
- **Ratio features**: HR/speed ratios, efficiency metrics
- **Interaction features**: Age-weight, age-height interactions
- **Polynomial features**: 2nd degree polynomial transformations
- **Domain-specific features**: HR reserve, speed efficiency, gain efficiency

### 🔄 **Data Augmentation**
- **TSAug integration**: Time warping, magnitude warping, jittering, scaling, rotation
- **Oversampling techniques**: SMOTE, ADASYN, random oversampling
- **Configurable augmentation**: Adjustable strength and probability parameters

### 🧹 **Advanced Preprocessing**
- **Feature scaling**: Standard, Robust, MinMax scaling
- **Feature selection**: Mutual information, F-regression, RFE
- **Outlier detection**: Isolation Forest, Local Outlier Factor, Elliptic Envelope

### 🤖 **Multiple ML Algorithms**
- **Random Forest**: Robust ensemble method
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Fast gradient boosting
- **Elastic Net**: Linear model with L1/L2 regularization
- **SVR**: Support Vector Regression

### 📊 **Comprehensive Evaluation**
- **Multiple metrics**: MAE, MSE, RMSE, R², MAPE
- **Feature importance**: Permutation importance, SHAP analysis
- **Residual analysis**: Distribution, normality tests
- **Visualization**: Actual vs predicted, residuals plots, feature importance

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd mock-vt1-vt2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Install TSAug for advanced data augmentation**:
```bash
pip install tsaug
```

## Quick Start

### 1. Basic Usage

Run the pipeline with default configuration:

```bash
python run_pipeline.py
```

### 2. Custom Configuration

Create your own configuration file and run:

```bash
python run_pipeline.py --config my_config.yaml
```

### 3. With Visualization

Run pipeline and generate evaluation plots:

```bash
python run_pipeline.py --create-plots
```

## Configuration

The pipeline is highly configurable through YAML configuration files. Key sections:

### Feature Engineering
```yaml
feature_engineering:
  demographics:
    include_bmi: true
    include_age_groups: true
    age_group_bins: [20, 30, 40, 50, 60, 70]
  statistical_features:
    include_percentiles: true
    percentiles: [10, 25, 50, 75, 90]
    include_skewness: true
    include_kurtosis: true
```

### Data Augmentation
```yaml
data_augmentation:
  enabled: true
  augmentation_factor: 3
  tsaug:
    time_warping:
      enabled: true
      probability: 0.5
      strength: 0.1
```

### Model Training
```yaml
model:
  algorithms:
    - "random_forest"
    - "xgboost"
    - "lightgbm"
    - "elastic_net"
    - "svr"
  hyperparameter_tuning:
    enabled: true
    method: "random_search"
    n_iterations: 50
```

## Data Format

### Input Data Structure
Your input CSV should contain the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `idx` | Sample index | Integer |
| `sid` | Subject ID | String |
| `gender` | Gender (M/F) | String |
| `age` | Age in years | Float |
| `weight` | Weight in kg | Float |
| `height` | Height in cm | Float |
| `hr_rest` | Resting heart rate | Float |
| `hr_mean` | Mean heart rate | Float |
| `hr_std` | Heart rate std | Float |
| `speed_mean` | Mean speed | Float |
| `speed_std` | Speed std | Float |
| `gain_mean` | Mean gain | Float |
| `gain_std` | Gain std | Float |
| `vt1` | VT1 value (target) | Float |
| `vt2` | VT2 value (target) | Float |

### Example Data
```csv
idx,sid,gender,age,weight,height,hr_rest,hr_mean,hr_std,speed_mean,speed_std,gain_mean,gain_std,vt1,vt2
0,103,M,27,73.6,176,65,94,11,13.05,2.5,15.9,63.0,127,170
1,93,M,56,85.4,182,50,92,2,17.53,3.85,170.8,68.6,109,151
```

## Output Structure

The pipeline generates several outputs:

```
outputs/
├── data/
│   ├── processed/
│   │   ├── feature_engineered_data.csv
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── predictions.csv
├── models/
│   ├── vt1_model.joblib
│   └── vt2_model.joblib
├── reports/
│   ├── evaluation_report.txt
│   ├── feature_importance.csv
│   └── plots/
│       ├── vt1_evaluation.png
│       └── vt2_evaluation.png
└── logs/
    └── pipeline.log
```

## Advanced Usage

### Programmatic Usage

```python
from mock_vt1_vt2.pipeline import VTPredictionPipeline

# Initialize pipeline
pipeline = VTPredictionPipeline('config/pipeline_config.yaml')

# Run complete pipeline
results = pipeline.run_pipeline()

# Access results
models = results['models']
evaluation_results = results['evaluation_results']

# Make predictions on new data
new_data = pd.read_csv('new_data.csv')
predictions = pipeline.predict(new_data)
```

### Custom Feature Engineering

```python
from mock_vt1_vt2.feature_engineering import FeatureEngineer

# Create custom configuration
config = {
    'demographics': {'include_bmi': True},
    'statistical_features': {'include_percentiles': True},
    # ... other settings
}

# Apply feature engineering
engineer = FeatureEngineer(config)
engineered_data = engineer.transform(raw_data)
```

### Model Evaluation

```python
from mock_vt1_vt2.evaluation import ModelEvaluator

# Evaluate models
evaluator = ModelEvaluator(config)
results = evaluator.evaluate_all_models(models, test_data)

# Generate plots
evaluator.create_evaluation_plots('outputs/plots/')

# Generate report
evaluator.generate_evaluation_report('outputs/report.txt')
```

## Experimentation

### Trying Different Configurations

1. **Feature Engineering Experiments**:
   - Enable/disable different feature types
   - Adjust polynomial degree
   - Modify statistical feature parameters

2. **Data Augmentation Experiments**:
   - Change augmentation factor
   - Adjust TSAug parameters
   - Try different oversampling methods

3. **Model Experiments**:
   - Add/remove algorithms
   - Adjust hyperparameter tuning
   - Modify cross-validation settings

### Example Configuration Variations

**Minimal Features**:
```yaml
feature_engineering:
  demographics:
    include_bmi: true
    include_gender_encoding: true
  statistical_features:
    include_percentiles: false
    include_skewness: false
    include_kurtosis: false
  polynomial_features:
    enabled: false
```

**Maximum Features**:
```yaml
feature_engineering:
  demographics:
    include_bmi: true
    include_age_groups: true
    include_gender_encoding: true
  statistical_features:
    include_percentiles: true
    include_skewness: true
    include_kurtosis: true
    include_entropy: true
    include_autocorrelation: true
  polynomial_features:
    enabled: true
    degree: 3
```

## Performance Tips

1. **For Large Datasets**:
   - Reduce augmentation factor
   - Use fewer algorithms
   - Disable expensive feature engineering

2. **For Better Performance**:
   - Enable all feature engineering
   - Use higher augmentation factor
   - Enable hyperparameter tuning

3. **For Faster Training**:
   - Use fewer CV folds
   - Reduce hyperparameter iterations
   - Disable expensive algorithms (SVR)

## Troubleshooting

### Common Issues

1. **TSAug Import Error**:
   ```bash
   pip install tsaug
   ```

2. **Memory Issues**:
   - Reduce augmentation factor
   - Use fewer features
   - Process in smaller batches

3. **Poor Performance**:
   - Check data quality
   - Enable more feature engineering
   - Try different algorithms

### Logging

The pipeline generates detailed logs in `logs/pipeline.log`. Check this file for:
- Feature engineering progress
- Model training status
- Error messages
- Performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{vt_prediction_pipeline,
  title={VT1/VT2 Prediction Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/mock-vt1-vt2}
}
```

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the example configurations
