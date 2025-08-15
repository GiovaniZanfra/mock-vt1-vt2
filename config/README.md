# Configuration Files

This directory contains configuration files for the project, following cookie cutter data science conventions.

## Files

### `dataset.yaml`

Main configuration file for dataset loading and processing.

#### Key Parameters

- **`root`**: Root directory containing the dataset (default: `data/raw`)
- **`v3_subfolder`**: Subfolder containing V3 data files (CSV files)
- **`metadata_subfolder`**: Subfolder containing metadata files (Parquet files)
- **`protocols`**: List of protocols to filter by (e.g., `["vt1", "vt2"]`)
- **`hr_columns`**: Heart rate column names to extract from CSV files
- **`speed_columns`**: Speed column names to extract from CSV files

#### Usage

```python
from mock_vt1_vt2.dataset import build_raw_dataset, _load_config

# Load configuration
config = _load_config("config/dataset.yaml")

# Build dataset
merged_metadata, raw_data = build_raw_dataset(config)
```

#### Customization

You can customize the configuration by:

1. **Changing protocols**: Modify the `protocols` list to include different protocol types
2. **Adding column names**: Add more column names to `hr_columns` or `speed_columns` if your CSV files use different naming conventions
3. **Adjusting paths**: Change the `root`, `v3_subfolder`, or `metadata_subfolder` paths to match your data structure

#### Example Configurations

**For VT1 and VT2 protocols only:**
```yaml
protocols:
  - "vt1"
  - "vt2"
```

**For different column naming:**
```yaml
hr_columns:
  - "heart_rate"
  - "HR"
  - "hr"

speed_columns:
  - "velocity"
  - "Speed"
  - "speed"
```

### `summarization.yaml`

Configuration file for timeseries summarization processing.

#### Key Parameters

- **`algorithm`**: Processing algorithm to use
  - `"ftp_cycling"`: Legacy FTP cycling algorithm
  - `"vo2max_cycling"`: Legacy VO2max cycling algorithm  
  - `"basic_moving_average"`: New basic moving average algorithm
- **`window_size`**: Window size for moving average calculations
- **`output_dir`**: Directory to save interim processed data
- **`algorithm_params`**: Algorithm-specific parameters

#### Usage

```python
from mock_vt1_vt2.timeseries_summarization import process_raw_to_interim

# Process raw data to interim features
output_path = process_raw_to_interim(
    raw_data_path="data/processed/processed_dataset.parquet",
    config_path="config/summarization.yaml"
)
```

#### Algorithm Examples

**FTP Cycling Algorithm:**
```yaml
algorithm: "ftp_cycling"
algorithm_params:
  verbose: false
  legacy:
    min_data_points: 30
    apply_filtering: true
    percentile: 0.75
```

**VO2max Cycling Algorithm:**
```yaml
algorithm: "vo2max_cycling"
algorithm_params:
  verbose: true
  legacy:
    min_data_points: 30
    apply_filtering: true
    percentile: 0.75
```

**Basic Moving Average Algorithm:**
```yaml
algorithm: "basic_moving_average"
window_size: 30
algorithm_params:
  window_size: 30
  include_percentiles: true
```

## Environment Variables

You can override config file paths using environment variables:

```bash
# Dataset config
export DATASET_CONFIG=/path/to/your/dataset.yaml

# Summarization config
export SUMMARIZATION_CONFIG=/path/to/your/summarization.yaml
```

## Running the Pipeline

### Dataset Loading
```bash
make data
```

### Timeseries Summarization
```bash
# Use default algorithm from config
make summarize

# Use specific algorithms
make summarize-ftp
make summarize-vo2max
make summarize-basic
```

### Complete Pipeline
```bash
# Run complete pipeline including summarization
make pipeline
```

This will:
1. Load the dataset using `config/dataset.yaml`
2. Process timeseries data using `config/summarization.yaml`
3. Generate features and models
4. Create visualizations
