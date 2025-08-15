from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from loguru import logger
import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path(
    "/home/g-brandao/workspace/crf-sandbox/experiments/at-ant/config/dataset.yaml"
)
MAXIMAL_FILENAMES = {"maximal", "maximal_bk"}


def list_metadata_parquets(root: Path, metadata_subfolder: str) -> list[Path]:
    p = root / metadata_subfolder
    if not p.exists():
        raise FileNotFoundError(f"Metadata folder not found: {p}")
    return sorted(list(p.glob("*.parquet")))


def load_all_metadata(meta_files: Sequence[Path]) -> pd.DataFrame:
    maximal_df: pd.DataFrame | None = None
    maximal_bk_df: pd.DataFrame | None = None
    metadata: pd.DataFrame | None = None

    for fp in meta_files:
        stem = fp.stem.lower()
        if stem == "maximal":
            maximal_df = pd.read_parquet(fp)
            continue
        if stem == "maximal_bk":
            maximal_bk_df = pd.read_parquet(fp)
            continue

        df = pd.read_parquet(fp)

        # incremental merge on "idx"
        if metadata is None:
            metadata = df.copy()
        else:
            metadata = metadata.merge(df, on="idx", how="outer", suffixes=("", f"_{stem}"))

    # validate presence
    if (metadata is None) or (maximal_bk_df is None) or (maximal_df is None):
        raise ValueError("Metadata not found, merged metadata files returned None.")

    maximal = maximal_df.merge(
        maximal_bk_df, how="outer", on="sid", suffixes=("_running", "_cycling")
    )
    metadata["sid"] = metadata["idx"].str.split("-").str[0]
    merged_metadata = metadata.merge(maximal, on="sid", how="left", suffixes=("", "_maximal"))
    return merged_metadata


def filter_by_protocol(meta: pd.DataFrame, protocol_list: Sequence[str]) -> pd.DataFrame:
    if not isinstance(protocol_list, list):
        protocol_list = list(protocol_list)
    meta_copy = meta.copy()
    meta_copy["protocol"] = meta_copy["idx"].str.split("-").str[1]
    filtered = meta_copy.query("protocol in @protocol_list").drop("protocol", axis=1)
    if filtered.empty:
        raise ValueError(f"Protocols {protocol_list} not found on metadata files.")
    return filtered


def extract_hr_from_ts(path: Path, hr_cols: Sequence[str]) -> List[float]:
    ts = pd.read_csv(path)
    for col in hr_cols:
        if col in ts:
            series = ts[col]
            try:
                values = pd.to_numeric(series, errors="coerce").tolist()
            except Exception:
                values = series.tolist()
            return values
    logger.info(
        f"Cols {hr_cols} not found on idx {path.stem}, returning empty array. Possible hr cols are {[col for col in ts.columns.tolist() if 'hr' in col]}"
    )
    return []


def extract_speed_from_ts(path: Path, speed_cols: Sequence[str]) -> List[float]:
    ts = pd.read_csv(path)
    for col in speed_cols:
        if col in ts:
            series = ts[col]
            try:
                values = pd.to_numeric(series, errors="coerce").tolist()
            except Exception:
                values = series.tolist()
            return values
    logger.info(
        f"Cols {speed_cols} not found on idx {path.stem}, returning empty array. Possible speed cols are {[col for col in ts.columns.tolist() if 'speed' in col]}"
    )
    return []


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_config(config_path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    cfg_path = (
        Path(config_path)
        if config_path
        else Path(os.environ.get("DATASET_CONFIG", DEFAULT_CONFIG_PATH))
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f) or {}
    logger.info(f"Loading config: {cfg_path}")
    return config


def build_raw_dataset(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds_cfg = config.get("dataset", config)  # allow top-level or nested under "dataset"

    root = Path(ds_cfg.get("root"))
    v3_subfolder = ds_cfg.get("v3_subfolder", "v3")
    metadata_subfolder = ds_cfg.get("metadata_subfolder", "metadata")
    protocols: list[str] = list(ds_cfg.get("protocols", []))
    hr_cols: list[str] = list(ds_cfg.get("hr_columns", []))
    speed_cols: list[str] = list(ds_cfg.get("speed_columns", []))

    meta_files = list_metadata_parquets(root, metadata_subfolder)
    logger.info(f"Found {len(meta_files)} metadata files.")
    merged_metadata = load_all_metadata(meta_files)
    if merged_metadata.empty:
        raise ValueError("No metadata files loaded. Verify .parquet files on the metadata folder")

    filtered_meta = filter_by_protocol(merged_metadata, protocols)
    logger.info(
        f"All metadata merged and filtered using {protocols}. {len(merged_metadata)} -> {len(filtered_meta)} rows."
    )

    v3_root = root / v3_subfolder
    if not v3_root.exists():
        raise FileNotFoundError(f"V3 folder not found: {v3_root}")

    idx_set = set(filtered_meta.idx.tolist())
    csv_paths = [path for path in v3_root.rglob("*.csv") if path.stem in idx_set]

    timeseries_data: list[dict[str, Any]] = []
    for path in csv_paths:
        entry: dict[str, Any] = {"idx": path.stem}
        entry["hr"] = extract_hr_from_ts(path, hr_cols)
        entry["speed"] = extract_speed_from_ts(path, speed_cols)
        timeseries_data.append(entry)

    timeseries_df = (
        pd.DataFrame(timeseries_data)
        if len(timeseries_data) > 0
        else pd.DataFrame(columns=["idx", "hr", "speed"])
    )

    raw_data = filtered_meta.merge(timeseries_df, how="left", on="idx")
    # Ensure missing lists are None so Parquet preserves nulls cleanly
    for col in ("hr", "speed"):
        if col in raw_data.columns:
            raw_data[col] = raw_data[col].apply(
                lambda x: (
                    list(x) if isinstance(x, (list, np.ndarray)) else (None if pd.isna(x) else x)
                )
            )

    logger.info(
        f"Raw data concluded with {len(raw_data)} rows. "
        f"{raw_data[['hr', 'speed']].dropna().shape[0]} rows have both speed and hr series."
    )
    return merged_metadata, raw_data


def main():
    """Main function to run dataset loading from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and process dataset using configuration file")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file (default: uses DATASET_CONFIG env var or default path)"
    )
    parser.add_argument(
        "--save", 
        action="store_true",
        help="Save processed dataset to data/processed/"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/processed",
        help="Output directory for processed data (default: data/processed)"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="processed_dataset.parquet",
        help="Output filename (default: processed_dataset.parquet)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = _load_config(args.config)
        
        logger.info("Starting dataset loading process...")
        
        # Build the raw dataset
        merged_metadata, raw_data = build_raw_dataset(config)
        
        logger.info(f"Successfully loaded dataset:")
        logger.info(f"  - Total metadata rows: {len(merged_metadata)}")
        logger.info(f"  - Filtered data rows: {len(raw_data)}")
        logger.info(f"  - Protocols used: {config.get('protocols', [])}")
        
        # Display some basic statistics
        if not raw_data.empty:
            hr_available = raw_data['hr'].notna().sum()
            speed_available = raw_data['speed'].notna().sum()
            both_available = (raw_data['hr'].notna() & raw_data['speed'].notna()).sum()
            
            logger.info(f"  - Rows with HR data: {hr_available}")
            logger.info(f"  - Rows with speed data: {speed_available}")
            logger.info(f"  - Rows with both HR and speed: {both_available}")
        
        # Save processed data if requested
        if args.save:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / args.output_file
            raw_data.to_parquet(output_path)
            logger.info(f"Saved processed dataset to: {output_path}")
        
        return raw_data, merged_metadata
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


if __name__ == "__main__":
    main()
