"""
Data ingestion pipeline for populating data/raw/ using external metadata + time series.
Refactored to follow cookie-cutter data science and configurable via mock_vt1_vt2.config.
"""

from pathlib import Path
from typing import List, Optional
import logging

import numpy as np
import pandas as pd

from .config import (
    DATA_SOURCE_ROOT,
    SOURCE_V3_SUBFOLDER,
    SOURCE_METADATA_SUBFOLDER,
    PROTOCOLS,
    HR_CANDIDATE_COLUMNS,
    SPEED_CANDIDATE_COLUMNS,
    MAXIMAL_FILENAMES,
    DATA_RAW,
    RAW_DATA_PARQUET_FILE,
    MERGED_METADATA_PARQUET_FILE,
)

logger = logging.getLogger(__name__)


def list_metadata_parquets(
    root: Path = DATA_SOURCE_ROOT,
    metadata_subfolder: str = SOURCE_METADATA_SUBFOLDER,
) -> List[Path]:
    """List available metadata parquet files from the external source."""
    p = root / metadata_subfolder
    if not p.exists():
        raise FileNotFoundError(f"Metadata folder not found: {p}")
    return sorted(list(p.glob("*.parquet")))


def load_all_metadata(meta_files: List[Path]) -> pd.DataFrame:
    """Load and merge all metadata parquet files.

    - Files named in MAXIMAL_FILENAMES are treated specially and merged on 'sid'.
    - Other files are merged successively on 'idx'.
    - Returns a single merged metadata DataFrame with a 'sid' column derived from 'idx'.
    """
    maximal_df = None
    maximal_bk_df = None
    metadata = None

    for fp in meta_files:
        stem = fp.stem.lower()
        if stem == "maximal":
            maximal_df = pd.read_parquet(fp)
            continue
        if stem == "maximal_bk":
            maximal_bk_df = pd.read_parquet(fp)
            continue

        df = pd.read_parquet(fp)

        # Incremental outer merge on 'idx'
        if metadata is None:
            metadata = df.copy()
        else:
            metadata = metadata.merge(df, on="idx", how="outer", suffixes=("", f"_{stem}"))

    if metadata is None or maximal_bk_df is None or maximal_df is None:
        raise ValueError("Metadata not found, merged metadata files returned None.")

    # Merge maximal info on 'sid'
    maximal = maximal_df.merge(maximal_bk_df, how="outer", on="sid", suffixes=("_running", "_cycling"))
    metadata["sid"] = metadata["idx"].str.split("-").str[0]
    merged_metadata = metadata.merge(maximal, on="sid", how="left", suffixes=("", "_maximal"))

    # Save for inspection
    MERGED_METADATA_PARQUET_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_metadata.to_parquet(MERGED_METADATA_PARQUET_FILE, index=False)
    logger.info(f"Saved merged metadata to {MERGED_METADATA_PARQUET_FILE}")

    return merged_metadata


def filter_by_protocol(meta: pd.DataFrame, protocol_list: Optional[List[str]] = None) -> pd.DataFrame:
    """Filter metadata by protocol codes present in 'idx' (after splitting by '-')."""
    if protocol_list is None:
        protocol_list = list(PROTOCOLS)
    meta_ = meta.copy()
    meta_["protocol"] = meta_["idx"].str.split("-").str[1]
    meta__filt = meta_.query("protocol in @protocol_list").drop("protocol", axis=1)
    if meta__filt.empty:
        raise ValueError(f"Protocols {protocol_list} not found on metadata files.")
    return meta__filt


def extract_hr_from_ts(path: Path, hr_cols: List[str]) -> np.ndarray:
    """Extract HR series from a time-series CSV by trying candidate columns in order."""
    ts = pd.read_csv(path)
    for col in hr_cols:
        if col in ts:
            return np.array(ts[col])
    logger.info(
        f"HR columns {hr_cols} not found for idx {path.stem}. Candidates present: {[c for c in ts.columns if 'hr' in c.lower()]}"
    )
    return np.array([])


def extract_speed_from_ts(path: Path, speed_cols: List[str]) -> np.ndarray:
    """Extract speed series from a time-series CSV by trying candidate columns in order."""
    ts = pd.read_csv(path)
    for col in speed_cols:
        if col in ts:
            return np.array(ts[col])
    logger.info(
        f"Speed columns {speed_cols} not found for idx {path.stem}. Candidates present: {[c for c in ts.columns if 'speed' in c.lower()]}"
    )
    return np.array([])


def merge_metadata_and_series(
    protocol_list: Optional[List[str]] = None,
    hr_cols: Optional[List[str]] = None,
    speed_cols: Optional[List[str]] = None,
    root: Path = DATA_SOURCE_ROOT,
    v3_subfolder: str = SOURCE_V3_SUBFOLDER,
    metadata_subfolder: str = SOURCE_METADATA_SUBFOLDER,
    keep_series_in_memory: bool = True,
) -> pd.DataFrame:
    """Merge metadata with HR and speed time series extracted from CSV files under v3/.

    Returns a DataFrame with columns including 'idx', 'sid', metadata fields, and 'hr', 'speed' arrays.
    """
    if protocol_list is None:
        protocol_list = list(PROTOCOLS)
    if hr_cols is None:
        hr_cols = list(HR_CANDIDATE_COLUMNS)
    if speed_cols is None:
        speed_cols = list(SPEED_CANDIDATE_COLUMNS)

    meta_files = list_metadata_parquets(root, metadata_subfolder)
    logger.info(f"Found {len(meta_files)} metadata files in {root / metadata_subfolder}.")
    meta = load_all_metadata(meta_files)
    if meta.empty:
        raise ValueError("No metadata files loaded. Verify .parquet files in the metadata folder")
    meta_filt = filter_by_protocol(meta, protocol_list)
    logger.info(
        f"All metadata merged and filtered using {protocol_list}. {len(meta)} -> {len(meta_filt)} rows."
    )

    v3_root = root / v3_subfolder
    if not v3_root.exists():
        raise FileNotFoundError(f"V3 folder not found: {v3_root}")

    # Collect available CSV series for selected idx
    selected_idx = set(meta_filt.idx.tolist())
    csv_paths = [path for path in v3_root.rglob("*.csv") if path.stem in selected_idx]
    logger.info(f"Found {len(csv_paths)} time series CSV files under {v3_root} for selected sessions.")

    timeseries_data = []
    for path in csv_paths:
        aux = {"idx": path.stem}
        hr_arr = extract_hr_from_ts(path, hr_cols)
        speed_arr = extract_speed_from_ts(path, speed_cols)
        # Convert to lists to ensure parquet compatibility
        aux["hr"] = hr_arr.tolist()
        aux["speed"] = speed_arr.tolist()
        if keep_series_in_memory:
            aux["_ts_path"] = str(path)
        timeseries_data.append(aux)

    timeseries_df = pd.DataFrame(timeseries_data)
    raw_data = meta_filt.merge(timeseries_df, on="idx", how="left")
    logger.info(
        f"Raw data concluded with {len(raw_data)} rows. {len(raw_data.dropna(subset=('hr', 'speed')))} rows have both speed and hr series."
    )
    return raw_data


def save_raw_data(raw_df: pd.DataFrame, output_path: Path = RAW_DATA_PARQUET_FILE) -> None:
    """Save raw combined data as parquet to preserve arrays."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(output_path, index=False)
    logger.info(f"Saved raw data to {output_path}")


def main():
    """Entry point to populate data/raw from the external source using configured parameters."""
    logger.info("Starting raw data ingestion pipeline...")
    raw_df = merge_metadata_and_series()
    save_raw_data(raw_df)
    logger.info("Raw data ingestion completed!")


if __name__ == "__main__":
    main()
