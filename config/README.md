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

## Environment Variables

You can override the config file path using the `DATASET_CONFIG` environment variable:

```bash
export DATASET_CONFIG=/path/to/your/config.yaml
```

## Running the Dataset Loader

Use the provided script to load your dataset:

```bash
python scripts/load_dataset.py
```

This will:
1. Load the configuration from `config/dataset.yaml`
2. Process the dataset according to your settings
3. Display statistics about the loaded data
4. Optionally save the processed dataset to `data/processed/`
